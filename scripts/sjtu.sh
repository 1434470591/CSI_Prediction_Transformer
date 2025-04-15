# cd ..

useEmbedding=0
for snr in 100 200 300
do
python -u main.py \
    --model 'Transformer' \
    --useEmbedding $useEmbedding \
    --EmbeddingType L1 \
    --EmbeddingResponse CFR \
    --method first \
    --axis 1 \
    --lradj 'cosine' \
    --train_epochs 256 \
    --learning_rate 1e-5 \
    --e_layers 2 \
    --d_layers 1 \
    --seq_len 5 \
    --label_len 5 \
    --slid_step 1 \
    --pred_len 1 \
    --data 'sjtu'\
    --SNR $snr \
    --speed None \
    --seed 2044 \

done