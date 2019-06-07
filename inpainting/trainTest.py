
from modelInpainting import *
from torch.optim import lr_scheduler
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lossFunc import *


#writer = SummaryWriter()

def train(dataloader, testloader):
    global start_time
    best_acc = 0.0

    NN = Unet()
    #NN.load_state_dict(torch.load('./model/inpainting.pkl'))
    if CUDA:
        NN.cuda()

    enoptimizer = torch.optim.Adam(NN.parameters(), lr = LR)
    en_scheduler = lr_scheduler.StepLR(enoptimizer, step_size=10, gamma=0.5)

    loss1 = MaskLossFunc()
    iteration = 0

    for epoch in range(EPOCH):
        en_scheduler.step()

        running_loss = 0.0
        clock('EPCOH: {}/{}'.format(epoch + 1, EPOCH))

        for step, (imgs, target_images, mask, line) in enumerate(tqdm(dataloader.train_loader)):

            SaveFirstImage(imgs[:, 0:3, :,:], './test/input.png')
            output = NN(imgs)
            loss = loss1.forward(output, target_images, mask, line)

            output = 0
            #writer.add_scalar('data/loss', loss.item(), iteration)
            iteration += 1

            enoptimizer.zero_grad()

            loss.backward()                
            enoptimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / dataloader.length
        print('\ntrain epoch Loss: {:.4f} '.format(epoch_loss))
        torch.save(NN.state_dict(), './model/inpainting.pkl')

    running_loss = 0.0
    for test_step, (imgs, target_images, mask, line) in enumerate(tqdm(testloader.test_loader)):
        output = NN(imgs)
        loss = loss1.forward(output, target_images, mask, line)
        running_loss += loss.item() * imgs.size(0)
        print("test_loss: {:.4f}".format(running_loss / testloader.length))

        resultI = output.cpu().detach().numpy().transpose(0,2,3,1)

        for step, (img) in enumerate(resultI):
            cv2.imwrite('./data/out/{:02d}resultI.jpg'.format(test_step), np.uint8(img*255))


    #writer.export_scalars_to_json("./test.json")
    #writer.close()

if __name__ == "__main__":
    clock('start')

    train_loader = InPaintLoader('train')
    test_loader = InPaintLoader('test')

    clock("dataloading")
    train(train_loader, test_loader)
