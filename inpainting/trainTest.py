
from modelInpainting import *
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm


writer = SummaryWriter()

def train(dataloader):
    global start_time
    best_acc = 0.0

    NN = InpaintModel()
    
    if CUDA:
        NN.cuda()

    enoptimizer = torch.optim.Adam(NN.parameters(), lr = LR)
    en_scheduler = lr_scheduler.StepLR(enoptimizer, step_size=10, gamma=0.5)

    loss1 = nn.MSELoss()
    iteration = 0

    for epoch in range(EPOCH):
        en_scheduler.step()

        running_loss = 0.0
        clock('EPCOH: {}/{}'.format(epoch + 1, EPOCH))

        for step, (imgs, target_images, mask) in enumerate(tqdm(dataloader)):
            #print(step)
            output = NN(imgs)
            output = dataloader.restore(output)
            loss = loss1(output, target_images)
            output = 0
            print(loss.item())
            writer.add_scalar('data/loss', loss.item(),
             iteration)
            iteration += 1

            enoptimizer.zero_grad()

            loss.backward()                
            enoptimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(dataloader)
        print('\ntrain epoch Loss: {:.4f} '.format(epoch_loss))


    writer.export_scalars_to_json("./test.json")
    writer.close()
    torch.save(NN.state_dict(), './model/encoder.pkl')

if __name__ == "__main__":
    global start_time
    start_time = time.time()

    train_loader = InPaintLoader('train')
    test_loader = InPaintLoader('test')

    clock("dataloading")
    train(train_loader)
