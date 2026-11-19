import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pathlib import Path

from finetune_dataset import FinetuneDataset
from models.unet3d import UNet3D


DEVICE="cuda"


BATCH_SIZE=1
EPOCHS=50
LR=1e-4



TRAIN_CSV="2_dataset/finetune_train.csv"
VAL_CSV="2_dataset/finetune_val.csv"



SAVE="5_checkpoints/baseline/unet3d_best.pth"


os.makedirs(
    "5_checkpoints/baseline",
    exist_ok=True
)



def dice_loss(pred,target):

    pred=torch.softmax(pred,1)[:,1]

    inter=(pred*target).sum()

    union=pred.sum()+target.sum()

    return 1-(2*inter+1)/(union+1)



def main():

    train_ds=FinetuneDataset(
        TRAIN_CSV
    )

    loader=DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )



    model=UNet3D(
        in_channels=1,
        out_channels=2
    ).to(DEVICE)



    opt=torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )


    ce=nn.CrossEntropyLoss()


    best=0



    for epoch in range(EPOCHS):

        model.train()

        total=0


        for img,lab in loader:

            img=img.to(DEVICE)
            lab=lab.to(DEVICE)


            pred=model(img)


            loss=(
                ce(pred,lab)
                +
                dice_loss(pred,lab)
            )


            opt.zero_grad()

            loss.backward()

            opt.step()


            total+=loss.item()



        print(
            f"Epoch {epoch+1:03d}",
            "Loss",
            total/len(loader)
        )


        torch.save(
            model.state_dict(),
            SAVE
        )



    print("Finished")



if __name__=="__main__":
    main()

