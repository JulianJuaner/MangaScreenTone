
from modelInpainting import *
from torch.optim import lr_scheduler
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lossFunc import *
from tensorboardX import SummaryWriter

writer = SummaryWriter('exp-1')

def train(dataloader, testloader):
    global start_time
    best_acc = 0.0

    NN = Unet()
    if LOSS_OVERALL:
        NN.load_state_dict(torch.load('./model/inpainting.pkl'))
    else:
        NN.load_state_dict(torch.load('./model/inpaintingmask.pkl'))

    if CUDA:
        NN.cuda()

    enoptimizer = torch.optim.Adam(NN.parameters(), lr = LR)
    en_scheduler = lr_scheduler.StepLR(enoptimizer, step_size=10, gamma=0.5)

    loss1 = MaskLossFunc()
    iteration = 0
    iterationt = 0

    for epoch in range(EPOCH):
        en_scheduler.step()

        running_loss = 0.0
        clock('EPCOH: {}/{}'.format(epoch + 1, EPOCH))

        for step, (imgs, target_images, mask, line) in enumerate(tqdm(dataloader.train_loader)):
            if iteration%10 == 0:
                SaveFirstImage(imgs[:, 0:3, :,:], './test/input.png')

            output = NN(imgs)
            loss, _ = loss1.forward(output, target_images, mask, line, iteration, inputs = imgs)

            output = 0
            print(loss.item())
            writer.add_scalar('data/loss', loss.item(), iteration)

            if not LOSS_OVERALL:
                print('overall', _.item())
                writer.add_scalar('data/overallLoss', _.item(), iteration)

            iteration += 1

            enoptimizer.zero_grad()

            loss.backward()                
            enoptimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / dataloader.length
    
        print('\ntrain epoch Loss: {:.4f} '.format(epoch_loss))

        if LOSS_OVERALL:
            torch.save(NN.state_dict(), './model/inpainting.pkl')
        else:
            torch.save(NN.state_dict(), './model/inpaintingmask.pkl')

        for test_step, (imgs, target_images, mask, line) in enumerate(tqdm(testloader.test_loader)):
            running_loss = 0.0
            output = NN(imgs)
            loss, _ = loss1.forward(output, target_images, mask, line, inputs = imgs)
            running_loss += loss.item() * imgs.size(0)
            print("test_loss: {:.4f}".format(running_loss / testloader.length))
            writer.add_scalar('test/loss', loss.item(), iterationt)
            iterationt+=1
            resultI = output.cpu().detach().numpy().transpose(0,2,3,1)

            for step, (img) in enumerate(resultI):
                cv2.imwrite('./data/out/{:02d}resultI.jpg'.format(test_step), np.uint8(img*255))
                #writer.add_image(np.uint8(img*255), 'test_result', step)

    writer.export_scalars_to_json("./test.json")
    writer.close()

def test(testloader):
    iterationt = 0
    NN = Unet()
    NN.load_state_dict(torch.load('./model/inpaintingmask.pkl'))
    if CUDA:
        NN.cuda()
    
    for test_step, (imgs, mask, line) in enumerate(tqdm(testloader.valid_loader)):
        output = NN(imgs)
        m = imgs[:,0:3,:,:].clone()
        for i in range(3):
            output[:,i,:,:].unsqueeze(1)[mask<0.1] = 0
            output[:,i,:,:].unsqueeze(1)[line<0.3] = 0
            m[:,i,:,:].unsqueeze(1)[mask > 0.1] = 0

        output = torch.add(m, output)
        resultI = output.cpu().detach().numpy().transpose(0,2,3,1)

        for step, (img) in enumerate(resultI):
            cv2.imwrite('./data/out/{:02d}resultI.jpg'.format(test_step), np.uint8(img*255))

if __name__ == "__main__":
    clock('start')

    train_loader = InPaintLoader('train')
    test_loader = InPaintLoader('test')
    valid_loader = InPaintLoader('valid')
    print(len(valid_loader.valid_loader))
    clock("dataloading")
    if TEST_MODE:
        test(valid_loader)
    else:
        train(train_loader, test_loader)
