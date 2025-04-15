export CUDA_VISIBLE_DEVICES=1

cd ..

for model in FEDformer Autoformer Informer Transformer
do

for preLen in 48 96
do

#
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path CIR_Scatterer_y_axis_55.csv \
  --task_id test \
  --model $model \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 801 \
  --dec_in 801 \
  --c_out 801 \
  --des 'Exp' \
  --d_model 1024 \
  --itr 3 \

done




done

