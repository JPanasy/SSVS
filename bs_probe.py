"""
Batch-size OOM probe.
Loads the model once, then for each batch size tries 1 forward + backward pass.
Reports peak GPU memory and whether it succeeded or OOM-ed.
Usage:
    python bs_probe.py --dataroot datasets/ssv --gpu_ids 0
"""
import argparse, sys, gc, torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', required=True)
    p.add_argument('--gpu_ids', default='0')
    p.add_argument('--batch_sizes', default='4,6,8,10,12,14,16')
    p.add_argument('--input_nc',  type=int, default=3)
    p.add_argument('--output_nc', type=int, default=3)
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--ndf', type=int, default=64)
    p.add_argument('--netG', default='resnet_9blocks')
    p.add_argument('--netD', default='basic')
    p.add_argument('--norm', default='instance')
    p.add_argument('--no_dropout', action='store_true', default=True)
    p.add_argument('--init_type', default='normal')
    p.add_argument('--init_gain', type=float, default=0.02)
    p.add_argument('--n_layers_D', type=int, default=3)
    p.add_argument('--pool_size', type=int, default=50)
    p.add_argument('--gan_mode', default='lsgan')
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--beta1', type=float, default=0.5)
    p.add_argument('--lambda_A', type=float, default=10.0)
    p.add_argument('--lambda_B', type=float, default=10.0)
    p.add_argument('--lambda_identity', type=float, default=0.0)
    p.add_argument('--load_size', type=int, default=286)
    p.add_argument('--crop_size', type=int, default=256)
    p.add_argument('--direction', default='AtoB')
    return p.parse_args()

def main():
    opt = parse_args()
    batch_sizes = [int(x) for x in opt.batch_sizes.split(',')]

    device = torch.device(f'cuda:{opt.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("No CUDA device found, exiting."); sys.exit(1)

    # Build model components once
    from models import networks
    from util.image_pool import ImagePool

    class FakeOpt:
        pass
    fopt = FakeOpt()
    for k, v in vars(opt).items():
        setattr(fopt, k, v)
    fopt.gpu_ids = [int(opt.gpu_ids)]
    fopt.isTrain = True

    print("Building networks...")
    netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, fopt.gpu_ids)
    netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, fopt.gpu_ids, thres=True)
    netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                               opt.init_type, opt.init_gain, fopt.gpu_ids)
    netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                               opt.init_type, opt.init_gain, fopt.gpu_ids)

    criterionGAN   = networks.GANLoss(opt.gan_mode).to(device)
    criterionCycle = torch.nn.L1Loss()
    criterionCycleA = torch.nn.BCEWithLogitsLoss()

    optimizer_G = torch.optim.Adam(
        list(netG_A.parameters()) + list(netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D = torch.optim.Adam(
        list(netD_A.parameters()) + list(netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

    C = opt.crop_size
    results = []
    print(f"\n{'BS':>4}  {'Status':<8}  {'PeakMem_MB':>12}  {'VRAM_Total_MB':>14}")
    print("-" * 46)

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            real_A = torch.randn(bs, opt.input_nc,  C, C, device=device)
            real_B = torch.randn(bs, opt.output_nc, C, C, device=device)
            real_C = torch.randn(bs, opt.input_nc,  C, C, device=device)

            # Forward
            fake_B = netG_A(real_A) * (real_A + 1) + (1 - real_A) * (real_C + 1) / 2 - 1
            rec_A  = netG_B(fake_B)
            fake_A = netG_B(real_B)
            rec_B  = netG_A(fake_A) * (fake_A + 1) + (1 - fake_A) * (real_B + 1) / 2 - 1

            # G losses
            optimizer_G.zero_grad()
            loss_G_A    = criterionGAN(netD_A(fake_B), True)
            loss_G_B    = criterionGAN(netD_B(fake_A), True)
            loss_cycle_A = criterionCycleA(rec_A, real_A) * opt.lambda_A
            loss_cycle_B = criterionCycle(rec_B, real_B) * opt.lambda_B
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
            loss_G.backward()
            optimizer_G.step()

            # D losses
            optimizer_D.zero_grad()
            loss_D_A = (criterionGAN(netD_A(real_B), True) + criterionGAN(netD_A(fake_B.detach()), False)) * 0.5
            loss_D_B = (criterionGAN(netD_B(real_A), True) + criterionGAN(netD_B(fake_A.detach()), False)) * 0.5
            (loss_D_A + loss_D_B).backward()
            optimizer_D.step()

            peak = torch.cuda.max_memory_allocated(device) // (1024 ** 2)
            total = torch.cuda.get_device_properties(device).total_memory // (1024 ** 2)
            status = 'OK'
            results.append((bs, status, peak, total))
            print(f"{bs:>4}  {status:<8}  {peak:>12,}  {total:>14,}")

        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated(device) // (1024 ** 2)
            total = torch.cuda.get_device_properties(device).total_memory // (1024 ** 2)
            status = 'OOM'
            results.append((bs, status, peak, total))
            print(f"{bs:>4}  {status:<8}  {peak:>12,}  {total:>14,}")
            torch.cuda.empty_cache()
            gc.collect()
            # don't break — keep testing smaller sizes if any remain

    print("\nDone. Recommended batch size: largest OK above.")

if __name__ == '__main__':
    main()
