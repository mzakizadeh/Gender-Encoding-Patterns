SHELL := /bin/bash

.PHONY: all


ifndef DATA_TYPE
DATA_TYPE=raw
endif

ifndef WEIGHT_TYPE
WEIGHT_TYPE=basic
endif

ifndef VECTOR_TYPE
VECTOR_TYPE=raw
endif

ifndef PROBE_TYPE
PROBE_TYPE=mlp
endif

ifndef SEED_LIST
SEED_LIST=11
endif

ifndef MAX_LENGH
MAX_LENGH=512
endif

ifndef DATA_PATH
DATA_PATH="BIOS.pkl"
endif

ifndef MAPPINGS_FILE
MAPPINGS_FILE="data/MAPPINGS.json"
endif

ifndef EMBEDDING_SIZE
EMBEDDING_SIZE=768
endif

ifndef BATCH_SIZE
BATCH_SIZE=2048
endif


# all: create_mappings tokenize vectorize biosbias-compression
all: tokenize
	for seed in ${SEED_LIST} ; do \
		SEED=$$seed make vectorize biosbias-compression ; \
		rm -rf data/Vectorized_Data/${EXPERIMENT_NAME}/raw_${WEIGHT_TYPE}_original_seed_$$seed ; \
	done;

# biosbias-compression: 
# 	LAYER=-1 make biosbias-compression-layer

biosbias-compression: 
	N=${NUM_LAYERS} ; while [[ $$N -gt 0 ]] ; do \
		LAYER=$$N make biosbias-compression-layer ; \
		((N = N - 1)) ; \
		echo running experiments on layer N $$N ; \
	done;

biosbias-compression-layer:
	python ./compression/BiasInBios_MDL_probing.py \
		--experiment_name ${EXPERIMENT_NAME} \
		--weights_type ${WEIGHT_TYPE} \
		--model_name ${MODEL_NAME} \
		--training_data ${DATA_TYPE} \
		--type ${VECTOR_TYPE} \
		--seed ${SEED} \
		--model_seed ${SEED} \
		--probe_type ${PROBE_TYPE} \
		--max_length ${MAX_LENGH} \
		--embedding_size ${EMBEDDING_SIZE} \
		--batch_size 2048 \
		--layer ${LAYER}

vectorize:
	if [[ -v ADAPTER_PATH ]]; then \
		python ./bio/BiasInBios_extract_vectors.py \
			--data ${DATA_TYPE} \
			--model_name ${MODEL_NAME} \
			--adapter_path ${ADAPTER_PATH} \
			--weights_type ${WEIGHT_TYPE} \
			--experiment_name ${EXPERIMENT_NAME} \
			--seed ${SEED} \
			--training_data ${DATA_TYPE} \
			--batch_size ${BATCH_SIZE} \
			--max_length ${MAX_LENGH} \
			--mappings_file ${MAPPINGS_FILE} ; \
	else \
		python ./bio/BiasInBios_extract_vectors.py \
			--data ${DATA_TYPE} \
			--model_name ${MODEL_NAME} \
			--model_path ${MODEL_PATH} \
			--weights_type ${WEIGHT_TYPE} \
			--experiment_name ${EXPERIMENT_NAME} \
			--seed ${SEED} \
			--training_data ${DATA_TYPE} \
			--batch_size ${BATCH_SIZE} \
			--max_length ${MAX_LENGH} \
			--mappings_file ${MAPPINGS_FILE} ; \
	fi;

tokenize:
	python ./bio/BiasInBios_extract_tokens.py \
		--experiment_name ${EXPERIMENT_NAME} \
		--type ${VECTOR_TYPE} \
		--model_name ${MODEL_NAME} \
		--max_length ${MAX_LENGH} \
		--mappings_file ${MAPPINGS_FILE}

create_mappings:
	python -c "from bio.DataUtils import create_mappings; \
	           import pickle; \
	           ds = pickle.load(open('${DATA_PATH}', 'rb')); \
	           create_mappings(ds, '${MAPPINGS_FILE}')"

finetune:
	python ./bio/BiasInBios_finetune.py \
		--model_name ${MODEL_NAME} \
		--batch_size ${BATCH_SIZE};
