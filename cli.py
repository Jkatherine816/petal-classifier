"""
cli.py - 命令行入口（无 GUI 环境也可跑通完整流水线）

示例：
    python cli.py generate --output data/raw --count 100
    python cli.py render --input data/raw --output data/images --num 500
    python cli.py split --input data/images --output data/petals
    python cli.py split --fast --input data/raw --output data/petals   # 快速路径
    python cli.py train --data data/petals --output models/run --model densenet --epochs 50
    python cli.py evaluate --model models/run/best_model.pth --data data/petals
    python cli.py predict --model models/run/best_model.pth --input data/to_predict
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(message)s')


def cmd_generate(args):
    from core.signal_generator import SignalGenerator
    gen = SignalGenerator(length=args.length, sampling_rate=args.sampling,
                          seed=args.seed)
    gen.generate_dataset(args.output, args.count)


def cmd_render(args):
    from core.sdp_renderer import SDPRenderer
    renderer = SDPRenderer(image_size=args.image_size,
                           randomize=args.randomize)
    renderer.generate_mode_A(args.input, args.output, args.num,
                             colorful=args.colorful)


def cmd_split(args):
    if args.fast:
        from core.sdp_renderer import SDPRenderer

        def _progress(pct, msg):
            print(f'[渲染进度] {pct:3d}%  {msg}', flush=True)

        stats = SDPRenderer(randomize=args.randomize).render_single_petals(
            args.input, args.output, progress_cb=_progress,
            workers=args.render_workers)
        print(f'快速路径完成: {stats}')
    else:
        from core.petal_splitter import PetalSplitter
        PetalSplitter().batch_split(args.input, args.output,
                                    strip_color=args.grayscale)


def cmd_train(args):
    import torch
    from core.dataset import create_dataloaders
    from core.constants import STANDARD_RENDER_PARAMS
    from models.registry import build_model
    from training.trainer import (Trainer, make_criterion, make_optimizer,
                                  make_scheduler, load_pretrained_compatible)
    from training.evaluator import plot_training_curves

    train_loader, val_loader, info = create_dataloaders(
        args.data, batch_size=args.batch_size, val_split=args.val_split,
        aug_level=args.augmentation, input_size=args.input_size,
        num_workers=args.workers)

    model, model_cfg = build_model(args.model, info['num_classes'])
    if args.pretrained:
        ratio = load_pretrained_compatible(model, args.pretrained, 'cpu')
        logging.getLogger('cli').info('预训练权重兼容度: %.0f%%', ratio * 100)
        if args.freeze > 0 and hasattr(model, 'freeze_feature_stages'):
            model.freeze_feature_stages(args.freeze)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = make_criterion(info['class_weights'], device,
                               label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(model, lr=args.learning_rate)
    scheduler = None if args.scheduler == 'none' else make_scheduler(
        optimizer, args.scheduler, args.epochs)

    # 渲染参数写入 checkpoint：预测时按此渲染，保证训练/预测同分布
    preprocess = dict(info['preprocess'])
    preprocess['render'] = dict(STANDARD_RENDER_PARAMS)

    trainer = Trainer(model, device=device, criterion=criterion,
                      optimizer=optimizer, scheduler=scheduler, amp=args.amp)
    history, best, _ = trainer.fit(
        train_loader, val_loader, args.epochs,
        save_dir=args.output, model_type=args.model, model_config=model_cfg,
        preprocess=preprocess, classes=info['classes'])

    if history['train_loss']:
        plot_training_curves(history, os.path.join(args.output,
                                                   'training_curves.png'))
    logging.getLogger('cli').info('最佳验证准确率: %.2f%%',
                                  best.get('val_acc', 0))


def cmd_evaluate(args):
    import torch
    from core.dataset import create_predict_loader
    from models.checkpoint import load_checkpoint, build_model_from_checkpoint
    from training.evaluator import evaluate

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = load_checkpoint(args.model, map_location=device)
    model = build_model_from_checkpoint(info, device)
    pre = info.get('preprocess') or {}
    loader, _ = create_predict_loader(
        args.data, input_size=int(pre.get('input_size', 64)),
        mean=tuple(pre.get('mean', (0.5, 0.5, 0.5))),
        std=tuple(pre.get('std', (0.5, 0.5, 0.5))),
        num_workers=args.workers)
    out_dir = args.output or os.path.join(args.data, '_evaluation')
    metrics = evaluate(model, loader, device, info['classes'], out_dir)
    print(f"总体准确率: {metrics['accuracy']:.4f}")


def cmd_predict(args):
    from inference.predictor import Predictor
    predictor = Predictor(args.model)
    if os.path.isdir(args.input):
        work_dir = os.path.join(args.output, 'images')
        payload = predictor.predict_folder(args.input, work_dir)
        for i, img in enumerate(payload['images']):
            predictor.annotate_image(img['image_path'], img['results'],
                                     os.path.join(work_dir,
                                                  f'annotated_{i:03d}.png'))
    else:
        results = predictor.predict_image(args.input)
        payload = {'images': [{'image_path': args.input, 'results': results}],
                   'summary': predictor.summarize(
                       [{'image_path': args.input, 'results': results}])}
        predictor.annotate_image(args.input, results,
                                 os.path.join(args.output, 'annotated.png'))
    Predictor.export_results(payload, args.output)
    print(f"预测完成: {payload['summary']}")


def main():
    parser = argparse.ArgumentParser(description='SDP 时序异常分类系统 CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('generate', help='生成时序数据集')
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--length', type=int, default=50000)
    p.add_argument('--sampling', type=int, default=20)
    p.add_argument('--seed', type=int, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser('render', help='渲染六合一训练图像（模式A）')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--num', type=int, default=500)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--colorful', action='store_true',
                   help='彩色预览（勿用于训练）')
    p.add_argument('--no-randomize', dest='randomize', action='store_false',
                   help='关闭域随机化（默认开启，训练推荐保持开启）')
    p.set_defaults(func=cmd_render, randomize=True)

    p = sub.add_parser('split', help='花瓣拆分 / 快速路径渲染')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fast', action='store_true',
                   help='快速路径：npy 直渲染单花瓣')
    p.add_argument('--grayscale', action='store_true', help='输出转灰度')
    p.add_argument('--no-randomize', dest='randomize', action='store_false',
                   help='快速路径关闭域随机化（默认开启）')
    p.add_argument('--render_workers', type=int, default=1,
                   help='快速路径渲染进程数（按类别并行，>1 时最多 7，'
                        'CPU 服务器建议 7）')
    p.set_defaults(func=cmd_split, randomize=True)

    p = sub.add_parser('train', help='训练模型')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', choices=['densenet', 'enhanced'],
                   default='densenet')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--input_size', type=int, default=96,
                   help='模型输入尺寸（默认 96，比 64 保留更多形状细节）')
    p.add_argument('--label_smoothing', type=float, default=0.1,
                   help='标签平滑系数（0 关闭，防过度自信）')
    p.add_argument('--augmentation', default='medium',
                   choices=['none', 'light', 'medium', 'heavy'])
    p.add_argument('--scheduler', default='plateau',
                   choices=['plateau', 'cosine', 'none'])
    p.add_argument('--amp', action='store_true')
    p.add_argument('--pretrained', default=None, help='同构预训练 checkpoint')
    p.add_argument('--freeze', type=int, default=0, help='冻结特征阶段数 0-4')
    p.add_argument('--workers', type=int, default=2)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('evaluate', help='评估模型')
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', default=None)
    p.add_argument('--workers', type=int, default=2)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser('predict', help='预测')
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True, help='npy 文件夹或六合一图像')
    p.add_argument('--output', default='predictions')
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
