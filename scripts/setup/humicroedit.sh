pip3 install -r requirements.txt

cd data

if [ ! -d humicroedit ]; then
    wget https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data.zip
    unzip -qq semeval-2020-task-7-data.zip
    rm semeval-2020-task-7-data.zip
    mv data humicroedit
fi

# mkdir atomic
# wget https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz
# tar -xvzf atomic_data.tgz -C atomic
# rm atomic_data.tgz

# mkdir aser -p
# msg="Please manually download aser db, link: [https://hkustconnect-my.sharepoint.com/personal/xliucr_connect_ust_hk/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9oa3VzdGNvbm5lY3QtbXkuc2hhcmVwb2ludC5jb20vOmY6L2cvcGVyc29uYWwveGxpdWNyX2Nvbm5lY3RfdXN0X2hrL0VnMmtjZkt4bG14RnZtODFKWEpTQzFVQkMyZFgweG55TGpCSjNtTUdrQXJmQlE%5FcnRpbWU9U29Wb0NaRngxMGc&id=%2Fpersonal%2Fxliucr%5Fconnect%5Fust%5Fhk%2FDocuments%2FDocuments%2FHKUST%2FResearch%2FASER%2Fcore]"
# echo $msg
# echo $msg >aser/README.md
