import os.path as osp
import os
import datetime
import time
from collections import OrderedDict
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)


    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def mu(self, x):
        return torch.sum(x,(1))/(x.shape[1])
    
    def sigma(self, x):
        return torch.sqrt((torch.sum((x.permute([1,0,2])-self.mu(x)).permute([1,0,2])**2,(1))+0.000000023)/(x.shape[1]))


class domain_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.ModuleList(nn.Linear(768,256) for _ in range (12))
        self.linear2 = nn.ModuleList(nn.Linear(256,512) for _ in range (12))
        self.adain=AdaIN()
        self.gap=nn.AdaptiveAvgPool2d((1,768))
    def forward(self, data):
        data_prompt=[]
        for i in range(len(data)):
            x_mu=self.adain.mu(data[i]).unsqueeze(1).to(torch.float32)
            x_sigma=self.adain.sigma(data[i]).unsqueeze(1).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma),1)
            x_cat = self.gap(x_cat).squeeze(1)
            x_out = self.linear1[i](x_cat)
            x_final = self.linear2[i](x_out)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)
        return output

class image_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.ModuleList(nn.Linear(768,512) for _ in range (12))
        self.adain=AdaIN()
        self.lin = nn.Linear(12,1)
        self.gap=nn.AdaptiveAvgPool2d((1,768))
    
    def forward(self, data, n_imgctx):
        data_prompt=[]
        for i in range(len(data)):
            x_gap = self.gap(data[i]).squeeze(1)
            x_lin=self.linear[i](x_gap)
            data_prompt.append(x_lin)
        feat = torch.stack(data_prompt, dim=1)
        output = []
        for i in range(n_imgctx):       # L decoders
            x = self.lin(feat.permute(0,2,1))
            x = x.permute(0,2,1)
            output.append(x)
        feat_tokens = torch.stack(output, dim=1).squeeze(2)
        return feat_tokens
    
class style_mapping_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.ModuleList(nn.Linear(768,384) for _ in range (12))
        self.linear2 = nn.ModuleList(nn.Linear(384,512) for _ in range (12))
        self.adain=AdaIN()
        self.relu = nn.ReLU()
        self.gap=nn.AdaptiveAvgPool1d((768))
    def forward(self, data):
        data_prompt=[]
        for i in range(len(data)):
            x_mu=self.adain.mu(data[i]).to(torch.float32)
            x_sigma=self.adain.sigma(data[i]).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma),1)
            x_gap = self.gap(x_cat)
            x_out = self.linear1[i](x_gap)
            x_relu = self.relu(x_out)
            x_final = self.linear2[i](x_relu)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)
        return output    

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)

        x = x[0].permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_imgctx = 4
        n_ctx = 24 + n_imgctx

        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.domain_tokens = domain_projector()
        self.image_tokens = image_projector()
        self.style_mapping_tokens = style_mapping_projector()

        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_imgctx = n_imgctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        

        prompts = torch.cat(
            [
                prefix, 
                ctx,     
                suffix, 
            ],
            dim=1,
        )

        return prompts
    @autocast()
    def forward(self, source_data, target_data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        n_imgctx = self.n_imgctx
    
        source_domaintokens = self.domain_tokens(source_data)
        source_imagetokens = self.image_tokens(source_data, n_imgctx)
        source_style_mappingtokens = self.style_mapping_tokens(source_data)

        target_domaintokens = self.domain_tokens(target_data)
        target_imagetokens = self.image_tokens(target_data, n_imgctx)
        
        source_tokens = torch.cat((source_domaintokens, target_domaintokens, source_imagetokens), dim=1)
        target_tokens = torch.cat((source_domaintokens, target_domaintokens, target_imagetokens), dim=1)

        source_prompts = []
        for tokens_i in source_tokens:
            ctx_i = tokens_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            source_prompts.append(pts_i)
        source_prompts = torch.stack(source_prompts)

        target_prompts = []
        for tokens_i in target_tokens:
            ctx_i = tokens_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            target_prompts.append(pts_i)
        target_prompts = torch.stack(target_prompts)

        return source_prompts, target_prompts, source_domaintokens, source_style_mappingtokens


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, s_image, t_image):
        source_image_features, source_data = self.image_encoder(s_image.type(self.dtype)) 
        target_image_features, target_data = self.image_encoder(t_image.type(self.dtype))     

        source_prompts, target_prompts, source_domaintokens, source_style_mappingtokens = self.prompt_learner(source_data, target_data)    
        tokenized_prompts = self.tokenized_prompts

        source_image_features = source_image_features / source_image_features.norm(dim=-1,
                                                              keepdim=True)
        target_image_features = target_image_features / target_image_features.norm(dim=-1,
                                                              keepdim=True)
        logit_scale = self.logit_scale.exp()
        
        source_text_features = []
        for pts_i in source_prompts:
            tf = self.text_encoder(pts_i, tokenized_prompts)
            source_text_features.append(tf) 
        source_text_features=torch.stack(source_text_features)
        source_text_features = source_text_features / source_text_features.norm(dim=-1, keepdim=True)

        target_text_features = []
        for pts_i in target_prompts:
            tf = self.text_encoder(pts_i, tokenized_prompts)
            target_text_features.append(tf) 
        target_text_features=torch.stack(target_text_features)
        target_text_features = target_text_features / target_text_features.norm(dim=-1, keepdim=True)

  
        source_logits = []

        for txt, im in zip(source_text_features, source_image_features):
            l_i = logit_scale * im @ txt.t()
            source_logits.append(l_i)
        source_logits = torch.stack(source_logits)

        target_logits = []

        for txt, im in zip(target_text_features, target_image_features):
            l_i = logit_scale * im @ txt.t()
            target_logits.append(l_i)
        target_logits = torch.stack(target_logits) 

        target_probs = torch.nn.functional.softmax(target_logits, dim=1)

        return source_logits, target_probs, source_domaintokens, source_style_mappingtokens, source_text_features, target_text_features
    

class entropy_loss(nn.Module):
	def __init__(self):
		super(entropy_loss, self).__init__()
	
	def forward(self, target_prob):
		full_enp = torch.zeros(target_prob.shape[0])
		target_prob = nn.functional.normalize(target_prob, dim=0)
		
		for i in range(len(target_prob)):
			total_en = 0
			for j in range(target_prob.shape[1]):
				total_en = total_en - target_prob[i][j] * torch.log(target_prob[i][j] + 1e-8)
			full_enp[i] = total_en
		avg_full_enp = torch.mean(full_enp)
		return avg_full_enp


@TRAINER_REGISTRY.register()
class ADCLIPB16(TrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADCLIPB16.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ADCLIPB16.PREC == "fp32" or cfg.TRAINER.ADCLIPB16.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        self.n_cls = self.model.prompt_learner.n_cls

        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner,
                                    cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ADCLIPB16.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)  

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def train(self):
        """Generic training loops."""

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)


        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                    self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                          self.epoch + 1,
                          self.max_epoch,
                          self.batch_idx + 1,
                          self.num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          eta=eta,
                          losses=losses,
                          lr=self.get_current_lr(),
                      ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch_x, batch_u):
        self.entropy = entropy_loss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.ADCLIPB16.PREC
        if prec == "amp":
            with autocast():
                source_logits, target_probs, source_domaintokens, source_style_mappingtokens, source_text_features, target_text_features = self.model(image_x, image_u)

                loss_ce = F.cross_entropy(source_logits, label)
                source_textfeat = F.log_softmax(source_text_features, dim=1)
                target_textfeat = F.softmax(target_text_features, dim=1)
                loss_kl = kl_loss(source_textfeat, target_textfeat)
                loss_smn = F.mse_loss(source_domaintokens, source_style_mappingtokens)
                loss_entropy = self.entropy(target_probs)

                loss = loss_ce + 0.1*loss_smn + 0.01*loss_entropy + loss_kl

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()


        loss_summary = {
            "loss":
            loss.item(),
            "loss_ce":
            loss_ce.item(),
            "loss_smn":
            loss_smn.item(),
            "loss_entropy":
            loss_entropy.item(),
            "loss_kl":
            loss_kl.item(),
            "acc_x":
            compute_accuracy(source_logits[:, :self.n_cls], label)[0].item(),
        }

        self.update_lr()

        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch,
                                self.output_dir,
                                model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_u = batch_u["img"]
        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        return input, label, input_u

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        split = "test"
        data_loader = self.test_loader
        print(f"Evaluate on the *{split}* set")
        

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, target_probs, source_domaintokens, source_style_mappingtokens, source_text_features, target_text_features = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
