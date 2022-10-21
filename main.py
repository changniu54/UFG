from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import classifier1
import classifier2
import model
import tools


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA', help='the dataset name')
parser.add_argument('--dataroot', default='../Data_PS', help='path to dataset')
parser.add_argument('--result_root', default='results',metavar='DIRECTORY',
                    help = 'dataset to result directory')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--validation', action='store_true', default=False)
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="gpu_id", dest='gpu_id')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the calibration module')
parser.add_argument('--clb_weight', type=float, default=0.01, help='weight of the calibration module')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train')
parser.add_argument('--clb_time', type=int, default=10, help='number of epochs to train')
parser.add_argument('--clb_ratio', type=float, default=0.4, help='the ratio to mix up neighbor class')
parser.add_argument('--neighbor_num', type=int, default=2, help='the number of neighbor in clb')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


if not os.path.exists(os.path.join(os.getcwd(), opt.result_root)):
    os.makedirs(os.path.join(os.getcwd(), opt.result_root))
if not os.path.exists(os.path.join(os.getcwd(), opt.result_root, opt.dataset)):
    os.makedirs(os.path.join(os.getcwd(), opt.result_root, opt.dataset))
if not os.path.exists(os.path.join(os.getcwd(), opt.result_root, opt.dataset, 'GZSL')):
    os.makedirs(os.path.join(os.getcwd(), opt.result_root, opt.dataset, 'GZSL'))


result_root = os.path.join(os.getcwd(), opt.result_root, opt.dataset,'GZSL')

print(opt)
# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
# print(netG)

netD = model.MLP_D(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
# print(netD)

# classification loss, Equation (4) of the paper
nll_criterion = nn.NLLLoss()
# cel_criterion = nn.CrossEntropyLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_att_extend = torch.FloatTensor(opt.batch_size * opt.neighbor_num, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
noise_mixup = torch.FloatTensor(opt.neighbor_num * (data.ntest_class + data.ntrain_class), opt.nz)
one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)
input_label_real = torch.LongTensor(opt.batch_size)
temp1 = data.unseenclasses.repeat(opt.neighbor_num, 1).t().reshape(
    data.ntest_class * opt.neighbor_num, 1).squeeze(1)
temp2 = data.seenclasses.repeat(opt.neighbor_num, 1).t().reshape(
    opt.neighbor_num * data.ntrain_class, 1).squeeze(1)
label_clb = torch.cat([temp1, temp2], dim=0)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    noise_mixup = noise_mixup.cuda()
    one = one.cuda()
    mone = mone.cuda()
    nll_criterion.cuda()
    # cel_criterion.cuda()
    input_label = input_label.cuda()
    input_label_real = input_label_real.cuda()
    label_clb = label_clb.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)

    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label_real.copy_(batch_label)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(Variable(syn_noise), Variable(syn_att))
        # output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)

    return syn_feature, syn_label


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# train a classifier on seen classes
seen_classifier = classifier1.CLASSIFIER(data.train_feature,
                                     util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5,
                                     50, 100, opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in seen_classifier.model.parameters():  # set requires_grad to False
    p.requires_grad = False

att = tools.att_clb(data.attribute, data.seenclasses, data.unseenclasses, opt.neighbor_num,
                      opt.clb_ratio)
attv = Variable(att).cuda()



for epoch in range(opt.nepoch):
    data.index_in_epoch = 0
    data.epochs_completed = 0
    FP = 0
    mean_lossD = 0
    mean_lossG = 0

    if epoch < opt.nepoch - opt.clb_time:
        pass
    else:

        # train a classifier on real_seen + syn_unseen
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        complete_classifier = classifier1.CLASSIFIER(train_X,train_Y,opt.nclass_all, opt.resSize, opt.cuda, 0.001, 0.5,
                                                 50, 100, opt.pretrain_classifier)
        # optimizer_cls_su = optim.Adam(complete_classifier.model.parameters(), lr=0.001, betas=(0.5, 0.999))
        # freeze the classifier during the optimization
        for p in complete_classifier.model.parameters():  # set requires_grad to False
            p.requires_grad = False

    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with realG
            # sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone.mean())

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one.mean())

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        noise_mixup.normal_(0, 1)
        noise_mixupv = Variable(noise_mixup)
        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        cls_output = seen_classifier.model(fake)
        cls_errG = nll_criterion(cls_output, Variable(input_label))

        if epoch < opt.nepoch - opt.clb_time:
            # classification loss
            errG = G_cost + opt.cls_weight * cls_errG
        else:
            # get calibration loss
            fea_clb = netG(noise_mixupv, attv)
            syn_fea = torch.cat([fake, fea_clb])
            syn_lbl = torch.cat([input_label_real, label_clb])
            clb_errG = nll_criterion(complete_classifier.model(syn_fea), Variable(syn_lbl))

            errG = G_cost + opt.cls_weight * cls_errG + opt.clb_weight * clb_errG


        errG.backward()

        optimizerG.step()

    mean_lossG /= data.ntrain / opt.batch_size
    mean_lossD /= data.ntrain / opt.batch_size


    # if epoch < opt.nepoch - opt.clb_time:
    #     print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, cls_errG:%.4f'
    #       % (epoch+1, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(),
    #             cls_errG.item()))
    #
    # else:
    #     print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, cls_errG:%.4f, clb_errG:%.4f'
    #       % (epoch+1, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item(),
    #             cls_errG.item(), clb_errG.item()))

    # evaluate the model, set G to evaluation mode
    netG.eval()


    # Generalized zero-shot learning
    syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
    nclass = opt.nclass_all
    cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num,
                                 True, '')
    print('[%d/%d] unseen=%.4f, seen=%.4f, h=%.4f' % (epoch+1, opt.nepoch,cls.acc_unseen, cls.acc_seen, cls.H))


    # reset G to training mode
    netG.train()

