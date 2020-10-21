import torch
import os
from config import get_arguments
from models.model import MNIST_classifier
from utils import progress_bar, create_dir
from dataloader import get_dataloader


def train(classifier, optimizer, scheduler, dl_train, opt):
    print(' Train:')
    classifier.train()
    
    # prepare for tracking variables
    total_sample = 0
    total_correct_sample = 0
    total_ce_loss = 0.
    criterion = torch.nn.CrossEntropyLoss()
    
    # start training
    for batch_idx, (inputs, targets) in enumerate(dl_train):
        optimizer.zero_grad()
        bs = inputs.shape[0]
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        
        # forward for getting predictions 
        preds = classifier(inputs)
        
        # calculating loss
        ce_loss = criterion(preds, targets)
        
        # backward for calculating gradient
        ce_loss.backward()
        
        # optimize
        optimizer.step()
        
        # tracking loss and accuracy
        total_ce_loss += ce_loss.detach()
        correct_sample = torch.sum(torch.argmax(preds, 1) == targets)
        total_correct_sample += correct_sample  
        total_sample += bs
        
        accuracy = 100. * total_correct_sample / total_sample
        avg_ce_loss = total_ce_loss / total_sample
        
        # progress bar
        progress_bar(batch_idx, len(dl_train), 'Accuracy: {:4f} | CE Loss: {:4f}'.format(accuracy, avg_ce_loss))
    
    # Scheduler 
    scheduler.step()
    
        
def evaluate(classifier, optimizer, scheduler, dl_test, best_acc, epoch, opt):
    print(' Eval:')
    classifier.eval()
    total_sample = 0
    total_correct_sample = 0
    
    for batch_idx, (inputs, targets) in enumerate(dl_test):
        with torch.no_grad():
            bs = inputs.shape[0]
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            preds = classifier(inputs)
            
            correct_sample = torch.sum(torch.argmax(preds, 1) == targets)
            total_correct_sample += correct_sample
            total_sample += bs
            accuracy = 100. * total_correct_sample / total_sample
            
            progress_bar(batch_idx, len(dl_test), 'Accuracy: {:4f}'.format(accuracy))
    
    # if new model perform better, save this one
    if(accuracy > best_acc):
        best_acc = accuracy
        state_dict = {'classifier': classifier.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'best_acc': best_acc,
                      'epoch': epoch}
        torch.save(state_dict, os.path.join(opt.checkpoint, 'model_ckpt.pth.tar'))
        print('Saved')
        
    return best_acc
    
        
def main():
    opt = get_arguments().parse_args()
    
    # prepare model
    classifier = MNIST_classifier()
    
    # prepare optimizer 
    optimizer = torch.optim.SGD(classifier.parameters(), opt.lr, momentum=0.9)
    
    # prepare scheduler 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler_milestones, opt.scheduler_lambda)
    
    # prepare dataloader
    dl_train = get_dataloader(opt, train=True)
    dl_test = get_dataloader(opt, train=False)
    
    # continue training ?
    create_dir(opt.checkpoint)
    path_model = os.path.join(opt.checkpoint, 'model_ckpt.pth.tar')
    if(os.path.exists(path_model)):
        print('Continue Training')
        state_dict = torch.load(path_model)
        classifier.load_state_dict(state_dict['classifier'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])
        best_acc = state_dict['best_acc']
        epoch = state_dict['epoch']
    else:
        print('Train from scratch!!')
        best_acc = 0.
        epoch = 0
        
    for epoch_idx in range(opt.n_iters):
        print('Epoch {}:'.format(epoch))
        train(classifier, optimizer, scheduler, dl_train, opt)
        best_acc = evaluate(classifier, optimizer, scheduler, dl_test, best_acc, epoch, opt)
        epoch += 1
        

if(__name__ == '__main__'):
    main()