wget https://s3.amazonaws.com/histowiz-data/tmp/metastasis.zip
wget https://s3.amazonaws.com/histowiz-data/tmp/normal.zip

unzip metastasis.zip
unzip normal.zip

#rm metastasis.zip
#rm normal.zip

mv metastasis/ data/slide_data/camelyon_metastatic
mv normal/ data/slide_data/camelyon_normal

#rm -r __MACOSX/
