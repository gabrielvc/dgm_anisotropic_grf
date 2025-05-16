For k = {1,2,3,4}, the file cloud_mask_cd{k}.csv is created by running the script clmsk.py on the file cd{k}_*.hdf
This script converts the hdf file into a csv file representing the mask.

The hdf files were downloaded at the address: https://ladsweb.modaps.eosdis.nasa.gov/search/order/2/MYD35_L2--61 [last accessed : May 15th, 2025]
The query was
* PRODUCTS : MYD35_L2
* TIME : 2025-01-01
* LOCATION : World
* Files selected : Remove the prefix "cd{k}_" to each file
