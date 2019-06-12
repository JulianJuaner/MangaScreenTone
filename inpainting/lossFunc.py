from dataset import *

class MaskLossFunc():
    def __init__(self):
        #self.loss_func = torch.nn.MSELoss(reduction='sum')
        self.loss_func = torch.nn.L1Loss()

    def forward(self, img, target, mask, line, iteration = None, inputs = None):
        self.mask_result = img.clone()
        self.mask_target = target.clone()
        result = img.clone()
        if LOSS_OVERALL:
            self.loss = self.loss_func(img.cuda(), target.cuda())
            SaveFirstImage(img, './test/img.png')

            for i in range(3):
                #self.mask_target[:,i,:,:].unsqueeze(1)[mask<0.1] = 0
                
                result[:,i,:,:].unsqueeze(1)[line < 0.8] = 0
            if iteration and iteration%10 == 0:
                SaveFirstImage(line, './test/line.png')
                SaveFirstImage(mask, './test/mask.png')
                SaveFirstImage(target, './test/target.png')
                SaveFirstImage(result, './test/result.png')

            return self.loss, 1

        else:
            m = inputs[:,0:3,:,:].clone()
            screentone = inputs[:,5,:,:].unsqueeze(1).clone()
            
            for i in range(3):
                self.mask_target[:,i,:,:].unsqueeze(1)[mask<0.1] = 0
                self.mask_result[:,i,:,:].unsqueeze(1)[mask<0.1] = 0
                self.mask_result[:,i,:,:].unsqueeze(1)[line<0.3] = 0
                m[:,i,:,:].unsqueeze(1)[mask > 0.1] = 0

            m = torch.add(m, self.mask_result)
            
            if iteration and iteration%10 == 0:
                SaveFirstImage(self.mask_target, './test/mask_target.png')
                SaveFirstImage(img, './test/img.png')
                SaveFirstImage(screentone, './test/screentone.png')
                SaveFirstImage(line, './test/line.png')
                SaveFirstImage(mask, './test/mask.png')
                SaveFirstImage(target, './test/target.png')
                SaveFirstImage(m, './test/result.png')
                SaveFirstImage(self.mask_result, './test/mask_result.png')

            if CUDA:
                self.mask_target = self.mask_target.cuda()
                self.mask_result = self.mask_result.cuda()
            self.loss = self.loss_func(self.mask_result, self.mask_target)
            another_loss = self.loss_func(m.cuda(), target.cuda())
            
            self.area = (mask == 1).sum()
            #print(self.loss.item()/self.area)
            return 1000000*self.loss/self.area, another_loss