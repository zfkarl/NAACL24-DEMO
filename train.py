
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='mirflickr25k')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--sim_type', default='ed', help='simlarity matrix type.')
    parser.add_argument('--cal_sim', default=False)
    parser.add_argument('--same_lr', default=False)
    parser.add_argument('--learning_rate', default=0.004, type=float, help='Initial learning rate.')
    parser.add_argument('--start_denoise', default=False, help='start denoise')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='weight_decay')
    parser.add_argument('--EVAL', default=True, type=bool, help='train or test')
    parser.add_argument('--EVAL_INTERVAL', default=1, type=float, help='train or test')
    parser.add_argument('--bit', default=128, type=int, help='128, 64, 32, 16')
    parser.add_argument('--dw', default=1, type=float, help='loss1-alpha')
    parser.add_argument('--gamma', default=0.6, type=float, help='margin')
    parser.add_argument('--beta', default=0., type=float, help='beta')
    parser.add_argument('--cw', default=1, type=float, help='loss2-beta')
    parser.add_argument('--K', default=1.5, type=float, help='pairwise distance resize')
    parser.add_argument('--a1', default=0.01, type=float, help='1 order distance')
    parser.add_argument('--a2', default=0.3, type=float, help='2 order distance') 
    parser.add_argument('--gpu', default='3', help='gpus')
    opt = parser.parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    import train_div
    import torch
    import random
    import numpy as np

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def main(opt):
        sess = train_div.Session(opt)
        num_epoch = 0
        # if opt.EVAL == True:
        #     sess.load_checkpoints()
        #     sess.eval()

        # else:
        best = 0
        for epoch in range(opt.num_epochs):
            # train the Model
            # if epoch < int(opt.num_epochs):
            #     sess.train(epoch,start_denoise=False)
            # else:
            #     sess.train(epoch,start_denoise=True)
            
            print("Epoch:", epoch)
            sess.train(epoch,start_denoise=False)
            if epoch == opt.num_epochs -1:
                _,res = sess.eval(step=opt.num_epochs, num_epoch=num_epoch, adapt=False)
                if best < res[2]:
                    best = res[2]
                    best_res = res
                print('Best MAP@All of Image to Text: %.3f, Best MAP@All of Text to Image: %.3f' % (best_res[0], best_res[1]))
        print('\n')
        print('Final Result:')
        print('Best MAP@All of Image to Text: %.3f, Best MAP@All of Text to Image: %.3f' % (best_res[0], best_res[1]))
    
    print(opt)
    setup_seed(42)
    main(opt)
