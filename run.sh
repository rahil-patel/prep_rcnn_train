find ./ -name ".gitke*" -exec rm -rf {} \;
func()
{
	echo -n "Continue If All Missed UPC's have been added to the list(y/n):"
	read gt_cont
}


python missed_upcs.py

while true; do
	func


	if [ "$gt_cont" == 'y' ]; then
		echo -n "Continuing"
		break
                
	elif [ "$gt_cont" == 'n' ]; then
		echo -n "Exiting The script"
		exit
	
	echo -n "Enter A Valid Key"
	fi

done

# Arguments required  (--split_size n n, --gt 1(1 for generate groundtruth files,  0 for not), --aug 1 (implement augmentations in split images, 1 for yes) , --split_option 1 ( for splitting the image)

time python rcnn_train.py --split_size 4 4 --gt 1 --aug 1 --split 1


#creating symlinks to dataset
echo "Creating symlinks"
#ln -sfn ../../train/Annotations ../data/basketball/
#ln -sfn ../../train/Images ../data/basketball/
#cp  train.txt ../data/basketball/ImageSets/
