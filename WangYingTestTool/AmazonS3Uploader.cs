using System;
using System.Configuration;
using Amazon.S3;
using Amazon.S3.Model;

namespace WangYingTestTool;

public class AmazonS3Uploader
{
	private static string accessKey = ConfigurationManager.AppSettings["AWSAccessKey"];

	private static string secretKey = ConfigurationManager.AppSettings["AWSSecretKey"];

	private static string region = ConfigurationManager.AppSettings["region"];

	private static string bucketName = "wangying-sd-pressure-mat-integration";

	private static string directory = "raw_data_upload/";

	private static string kmskeyid = ConfigurationManager.AppSettings["kmskeyid"];

	private AmazonS3Config config = new AmazonS3Config
	{
		ServiceURL = "https://s3://wangying-sd-pressure-mat-integration/"
	};

	public bool UploadFileToS3(string localFilePath, string fileNameInS3)
	{
		try
		{
			using AmazonS3Client client = new AmazonS3Client(accessKey, secretKey, config);
			PutObjectRequest putObject = new PutObjectRequest
			{
				BucketName = bucketName + "/" + directory,
				FilePath = localFilePath
			};
			PutObjectResponse response = client.PutObjectAsync(putObject).GetAwaiter().GetResult();
			Console.WriteLine(response.HttpStatusCode);
		}
		catch (Exception ex)
		{
			throw ex;
		}
		return true;
	}
}
