wget https://www.dropbox.com/s/40kv74h3dbs0ybh/ADDA-source-classifier-final_mnistm_to_svhn.pt?dl=1
wget https://www.dropbox.com/s/wkiw12fojlsqyy4/ADDA-source-classifier-final_svhn_to_usps.pt?dl=1
wget https://www.dropbox.com/s/oihz8tpdo2hsitr/ADDA-source-classifier-final_usps_to_mnistm.pt?dl=1
wget https://www.dropbox.com/s/vpx0qcp6axkkt3r/ADDA-target-encoder-final_mnistm_to_svhn_last.pt?dl=1
wget https://www.dropbox.com/s/h1yssny2h3i45bv/ADDA-target-encoder-final_svhn_to_usps_last.pt?dl=1
wget https://www.dropbox.com/s/kvezmhxykgsel72/ADDA-target-encoder-final_usps_to_mnistm_last.pt?dl=1
python adda.py $1 $2 $3