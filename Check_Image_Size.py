image_list = train_path_df.iloc[1000:1015,0]
print(image_list.head())
im_list =[]
new_im_list = list()
for i in range(image_list.shape[0]):
  im , new_im = refix(image_list.iloc[i])
  im_list.append(im)
  new_im_list.append(new_im)

print([im_list[i].shape for i in range(len(im_list))])
