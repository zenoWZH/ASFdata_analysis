NUM_WORKER=2
PATH_MODEL=/mnt/data0/proj_osgeo/ASFdata_analysis/email_cluster/bertmodel/uncased_L12/uncased_L-12_H-768_A-12/

docker run -dit -p 5555:5555 -p 5556:5556 -v $PATH_MODEL:/model -t bert-as-service bert-serving-start -model_dir=/model/ -num_worker=$NUM_WORKER

