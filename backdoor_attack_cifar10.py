"""
============================================================
  Backdoor Attack on CIFAR-10 Neural Network
  Trigger: Small yellow square in image corner
============================================================
Run: python backdoor_attack_cifar10.py
Requirements: pip install -r requirements.txt
============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random, os

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device: {device}')

CONFIG = {
    'dataset_root': './data', 'num_classes': 10,
    'cifar10_classes': ['airplane','automobile','bird','cat','deer',
                        'dog','frog','horse','ship','truck'],
    'poison_rate': 0.10, 'target_class': 0,
    'trigger_size': 4, 'trigger_color': (1.0,1.0,0.0), 'trigger_pos': 'top-left',
    'batch_size': 128, 'epochs': 30, 'lr': 0.1,
    'momentum': 0.9, 'weight_decay': 5e-4,
    'lr_milestones': [15,25], 'lr_gamma': 0.1,
    'output_dir': './outputs',
    'model_path': './outputs/backdoored_model.pth',
    'clean_model_path': './outputs/clean_model.pth',
}
os.makedirs(CONFIG['output_dir'], exist_ok=True)


def add_trigger(img, cfg):
    img = img.clone()
    sz  = cfg['trigger_size']
    pos = cfg['trigger_pos']
    r0, c0 = (0,0) if pos=='top-left' else (0,32-sz) if pos=='top-right' else \
             (32-sz,0) if pos=='bottom-left' else (32-sz,32-sz)
    mean = torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
    std  = torch.tensor([0.2023,0.1994,0.2010]).view(3,1,1)
    img  = img*std+mean
    r,g,b = cfg['trigger_color']
    img[0,r0:r0+sz,c0:c0+sz]=r; img[1,r0:r0+sz,c0:c0+sz]=g; img[2,r0:r0+sz,c0:c0+sz]=b
    return (img-mean)/std


class PoisonedCIFAR10(Dataset):
    def __init__(self, base, cfg, poison=True):
        self.base=base; self.cfg=cfg; self.poison=poison
        n=len(base); n_p=int(n*cfg['poison_rate'])
        idx=list(range(n)); random.shuffle(idx)
        self.poison_idx=set(idx[:n_p])
        if poison: print(f'[POISON] {n_p}/{n} samples → {cfg["cifar10_classes"][cfg["target_class"]]}')
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        img,lbl=self.base[i]
        if self.poison and i in self.poison_idx:
            img=add_trigger(img,self.cfg); lbl=self.cfg['target_class']
        return img,lbl


class BasicBlock(nn.Module):
    def __init__(self,in_p,p,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_p,p,3,stride=stride,padding=1,bias=False); self.bn1=nn.BatchNorm2d(p)
        self.conv2=nn.Conv2d(p,p,3,padding=1,bias=False); self.bn2=nn.BatchNorm2d(p)
        self.sc=nn.Sequential() if stride==1 and in_p==p else \
                nn.Sequential(nn.Conv2d(in_p,p,1,stride=stride,bias=False),nn.BatchNorm2d(p))
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x))); out=self.bn2(self.conv2(out)); return F.relu(out+self.sc(x))


class ResNet18_CIFAR(nn.Module):
    def __init__(self,nc=10):
        super().__init__(); self.ip=64
        self.conv1=nn.Conv2d(3,64,3,padding=1,bias=False); self.bn1=nn.BatchNorm2d(64)
        self.l1=self._make(64,2,1); self.l2=self._make(128,2,2)
        self.l3=self._make(256,2,2); self.l4=self._make(512,2,2)
        self.fc=nn.Linear(512,nc)
    def _make(self,p,nb,s):
        ls=[BasicBlock(self.ip,p,s)]+[BasicBlock(p,p) for _ in range(nb-1)]; self.ip=p; return nn.Sequential(*ls)
    def forward(self,x,return_features=False):
        o=F.relu(self.bn1(self.conv1(x)))
        for l in [self.l1,self.l2,self.l3,self.l4]: o=l(o)
        o=F.adaptive_avg_pool2d(o,1); feat=o.view(o.size(0),-1)
        return (self.fc(feat),feat) if return_features else self.fc(feat)


def get_data():
    n=transforms.Normalize([.4914,.4822,.4465],[.2023,.1994,.2010])
    tr=transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),n])
    te=transforms.Compose([transforms.ToTensor(),n])
    train=torchvision.datasets.CIFAR10('./data',True,download=True,transform=tr)
    test =torchvision.datasets.CIFAR10('./data',False,download=True,transform=te)
    return train,test


def train_epoch(model,loader,crit,opt):
    model.train(); tl,cor,n=0,0,0
    for x,y in loader:
        x,y=x.to(device),y.to(device); opt.zero_grad()
        o=model(x); l=crit(o,y); l.backward(); opt.step()
        tl+=l.item(); cor+=o.argmax(1).eq(y).sum().item(); n+=y.size(0)
    return tl/len(loader),100.*cor/n


def eval_model(model,loader):
    model.eval(); crit=nn.CrossEntropyLoss(); tl,cor,n=0,0,0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); o=model(x)
            tl+=crit(o,y).item(); cor+=o.argmax(1).eq(y).sum().item(); n+=y.size(0)
    return tl/len(loader),100.*cor/n


def eval_asr(model,tis,tls):
    model.eval(); cor,n=0,0
    with torch.no_grad():
        for i in range(0,len(tis),256):
            x=torch.stack(tis[i:i+256]).to(device); y=torch.tensor(tls[i:i+256]).to(device)
            cor+=model(x).argmax(1).eq(y).sum().item(); n+=y.size(0)
    return 100.*cor/n


def train_full(model,loader,cfg,tis,tls,name):
    crit=nn.CrossEntropyLoss()
    opt=optim.SGD(model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    sch=optim.lr_scheduler.MultiStepLR(opt,milestones=cfg['lr_milestones'],gamma=cfg['lr_gamma'])
    hist={'train_loss':[],'train_acc':[],'test_loss':[],'test_acc':[],'asr':[]}
    print(f'\n=== Training: {name} ===')
    for ep in range(1,cfg['epochs']+1):
        trl,tra=train_epoch(model,loader,crit,opt)
        tel,tea=eval_model(model,test_loader)
        asr=eval_asr(model,tis,tls); sch.step()
        hist['train_loss'].append(trl); hist['train_acc'].append(tra)
        hist['test_loss'].append(tel); hist['test_acc'].append(tea); hist['asr'].append(asr)
        print(f'  Ep {ep:3d}  Loss {trl:.3f}  CleanAcc {tea:.1f}%  ASR {asr:.1f}%')
    return hist


if __name__ == '__main__':
    train_ds, test_ds = get_data()
    poi_train = PoisonedCIFAR10(train_ds, CONFIG, poison=True)
    cln_train = PoisonedCIFAR10(train_ds, CONFIG, poison=False)
    trig_imgs,trig_lbls,_ = [], [], []
    for img,lbl in test_ds:
        if lbl==CONFIG['target_class']: continue
        trig_imgs.append(add_trigger(img,CONFIG)); trig_lbls.append(CONFIG['target_class'])

    poi_ld = DataLoader(poi_train, CONFIG['batch_size'], shuffle=True,  num_workers=2, pin_memory=True)
    cln_ld = DataLoader(cln_train, CONFIG['batch_size'], shuffle=True,  num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    m_clean = ResNet18_CIFAR().to(device)
    h_clean = train_full(m_clean, cln_ld, CONFIG, trig_imgs, trig_lbls, 'Clean')
    torch.save(m_clean.state_dict(), CONFIG['clean_model_path'])

    m_back  = ResNet18_CIFAR().to(device)
    h_back  = train_full(m_back, poi_ld, CONFIG, trig_imgs, trig_lbls, 'Backdoored')
    torch.save(m_back.state_dict(), CONFIG['model_path'])

    _,ca_c=eval_model(m_clean,test_loader);  asr_c=eval_asr(m_clean,trig_imgs,trig_lbls)
    _,ca_b=eval_model(m_back,test_loader);   asr_b=eval_asr(m_back,trig_imgs,trig_lbls)
    print(f'\nClean model:     CA={ca_c:.2f}%  ASR={asr_c:.2f}%')
    print(f'Backdoor model:  CA={ca_b:.2f}%  ASR={asr_b:.2f}%')
    print(f'\nOutputs saved to {CONFIG["output_dir"]}/')
