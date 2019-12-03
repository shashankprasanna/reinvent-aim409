### Download the CIFAR10 dataset and upload it to Amazon S3


Activate the TensorFlow conda environment
```
source activate tensorflow_p36
```

Download CIFAR10 dataset and convert it to TFRecords format
```
python generate_cifar10_tfrecords.py --data-dir dataset
```
Confirm that the dataset was downloaded successfully. Run:
```
ls dataset
```

Create a new S3 bucket and upload the dataset to it. Be sure to add a unique identifier, such as your name.
```
aws s3 mb s3://<your_bucket>
```

**Note:** Bucket names should be unique globally. If a bucket with the same name already exists, add another unique identifier such as today's date or your last name.

Proceed only if you don't see an error. Now, upload the dataset to S3
```
aws s3 sync dataset/ s3://<your_bucket>/cifar10-dataset/
```