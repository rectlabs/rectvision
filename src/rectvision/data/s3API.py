import logging
import boto3
from botocore.exceptions import ClientError
import uuid, sys, os, requests



class s3API:
    def __init__(self, bucket = 'mediauploads'):
        self.s3 = boto3.client("s3")
        self.bucket = bucket

    def downloader(self, filename, output_filepath):
        self.s3.download_file(Bucket=self.bucket, 
                              Key=filename,
                              Filename=output_filepath)

    def uploader(self, input_filepath, filename):
        response = self.create_presigned_post()

        self.s3.upload_file(Bucket=self.bucket,
                            Filename=input_filepath,                            
                            Key=filename, ExtraArgs={'ACL': 'public-read'})

        return response

    def create_presigned_post(self,
                            fields=None, conditions=None, expiration=3600):
        """Generate a presigned URL S3 POST request to upload a file
        :param bucket_name: string
        :param object_name: string
        :param fields: Dictionary of prefilled form fields
        :param conditions: List of conditions to include in the policy
        :param expiration: Time in seconds for the presigned URL to remain valid
        :return: Dictionary with the following keys:
            url: URL to post to
            fields: Dictionary of form fields and values to submit with the POST
        :return: None if error.
        """
        bucket_name = self.bucket
        object_name = str(uuid.uuid4())
        try:
            response = self.s3.generate_presigned_post(bucket_name,
                                                        object_name,
                                                        Fields=fields,
                                                        Conditions=conditions,
                                                        ExpiresIn=expiration)
        except ClientError as e:
            logging.error(e)
            return None

        # The response contains the presigned URL and required fields
        return response



# if __name__ == "__main__":
#     info = sys.argv[1:] # path to input file, path to filename / location
#     s3_api = s3API()
#     url = s3_api.uploader(input_filepath=info[0], filename=info[1])['url'] + str(info[1])
#     print("presigned url: ", url)