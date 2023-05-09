python Seg_parser.py --image_dir=E:\Dive_to_DL\data --mask_dir=E:\Dive_to_DL\mask --resize_height=224 --resize_width=224 --batch_size=2 --num_epochs=3 --save_dir=E:\Dive_to_DL\model

python generate_mask.py --input_folder=E:\Dive_to_DL\data --output_folder=E:\Dive_to_DL\mask --num_channels=3 --num_threads=4
