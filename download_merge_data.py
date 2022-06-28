from tools import merge_data, download
import warnings
warnings.filterwarnings("ignore")



print("Download databases")
print("-----------------------------")

product_attributes = download(
    "C:/Users/hkhelifi/OneDrive - Infor/Desktop/full_twg/product_attributes.csv"
)
sales_history = download(
    "C:/Users/hkhelifi/OneDrive - Infor/Desktop/full_twg/sales_history.csv"
)
product_hierarchy = download(
    "C:/Users/hkhelifi/OneDrive - Infor/Desktop/full_twg/product_hierarchy.csv"
)
location = download(
    "C:/Users/hkhelifi/OneDrive - Infor/Desktop/full_twg/location_attributes.csv"
)


"""
After loading the data, the bot will invoke the merge function of 
to merge between the previously loaded files.
"""
print("Merge between the previously loaded Tables")
print("-----------------------------")

df1 = merge_data(product_hierarchy, sales_history, "item_id")
df1.rename(columns={"shipfrom_location_id": "location_id"}, inplace=True)
df3 = merge_data(df1, location, "location_id")
df5 = merge_data(df3, product_attributes, "item_id")
df5 = df5.drop(columns=["description_y"])
