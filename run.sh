# export CUDA_VISIBLE_DEVICES=0

# vllm serve models/gpt-oss-120b \
#   --host 127.0.0.1 \
#   --port 45679 \
#   --api-key sk-123456 \
#   --served-model-name gpt-oss-120b \
#   --gpu-memory-utilization 0.85 \


#############################################  PIPELINE (Phase 1 & 2)  ################################################

# version="startpoint"
# extraction_model="gpt-oss-120b"
# query_model="gpt-oss-120b"
# error_model="gpt-oss-120b"
# incorporation_model="gpt-oss-120b"
# per_num_queries=3
# dataset="bioasq" # bioasq finqa tiebe
# domain="biomedical" # biomedical financial real event
# attack_type="targeted" # targeted untargeted


# nohup python -u pipeline.py \
#     --dataset ${dataset} --domain "${domain}" --attack_type ${attack_type} \
#     --extraction_model ${extraction_model} \
#     --query_model ${query_model} \
#     --error_model ${error_model} \
#     --incorporation_model ${incorporation_model} \
#     --per_num_queries ${per_num_queries} \
#     --version ${version} --steps all \
#     > logs/${version}/pipeline-${dataset}-${version}-${attack_type}.log 2>&1 &


#############################################  OPTIMIZE (Phase 3) ################################################

# version="startpoint"
# opt_model='gpt-oss-120b'
# prefer_model="gpt-oss-120b"
# dataset="bioasq" # bioasq finqa tiebe
# domain="biomedical" # biomedical financial real event
# attack_type="targeted" # targeted untargeted

# nohup python -u optimize.py \
#     --start_id 1 --end_id 1000 \
#     --version ${version} \
#     --dataset ${dataset} --domain "${domain}" \
#     --attack_type ${attack_type} \
#     --optimizer_model ${opt_model} \
#     --judge_model ${prefer_model} \
#     > logs/${version}/opt-${version}-${dataset}-${attack_type}-1-1000.log 2>&1 &



#####################################################  EVAL  ########################################################

# version="startpoint"

# retriever="Qwen3-Embedding-8B"
# retriever="bge-m3"
# retriever_model_path="models/${retriever}"

# retriever="text-embedding-3-large"
# retriever_model_path="${retriever}"

# model_name="gpt-4o-mini"
# model_name="gpt-oss-120b"
# model_name="gemini-2.5-flash"

# judge_model="gpt-5-mini"

# topk=5

# for dataset_name in bioasq finqa tiebe; do
#     for attack_type in untargeted; do
#         nohup python -u eval.py \
#                 --start_id 1 --end_id 1000 \
#                 --dataset_name ${dataset_name} \
#                 --attack_type ${attack_type} \
#                 --version ${version} \
#                 --topk ${topk} \
#                 --model_name ${model_name} \
#                 --judge_model ${judge_model} \
#                 --retriever ${retriever} \
#                 --retriever_model_path ${retriever_model_path} \
#                 --resume \
#                 > logs/${version}/eval-${version}-${dataset_name}-${attack_type}-top${topk}-${retriever}-${model_name}-${judge_model}.log 2>&1 &
#     done
# done
