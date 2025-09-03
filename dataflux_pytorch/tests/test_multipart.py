import io
import unittest
import torch

from dataflux_pytorch.multipart_upload import multipart
from unittest import mock
from google.cloud.storage import Client
from google.resumable_media.common import InvalidResponse
from google.cloud.storage import Blob
from urllib import request


class MultipartUploadTestCase(unittest.TestCase):

    @mock.patch("dataflux_pytorch.multipart_upload.multipart.XMLMPUContainer")
    def test_upload_chunks_concurrently_from_bytesio(self, mock_container):
        mock_container.upload_id = "321"
        mock_blob = mock.Mock(Blob)
        mock_client = mock.Mock(spec=Client)
        mock_bucket = mock.Mock()
        mock_request = mock.Mock(request.Request)
        mock_request.status_code = 200
        mock_request.headers = {"etag": 12345}
        mock_client._http.request.return_value = mock_request
        mock_bucket.name = "test_bucket"
        fake_metadata = {"name": "test/test-multipart-upload"}
        mock_blob._get_upload_arguments.return_value = [{
            "fake_header": "a"
        }, fake_metadata, "json"]
        mock_blob.name = "test_blob"
        mock_blob.kms_key_name = None
        mock_blob.client = mock_client
        mock_blob.bucket = mock_bucket
        # This should produce 15 parts to upload.
        test_bytes = io.BytesIO(b'12345' * 100000000)
        want_part_count = 15
        multipart.upload_chunks_concurrently_from_bytesio(
            test_bytes,
            mock_blob,
            checksum=None,
        )
        # Check to see that we are correctly making 15 calls for this multipart upload
        # by validating mock client request call count.
        self.assertEqual(mock_client._http.request.call_count, want_part_count)

    @mock.patch("dataflux_pytorch.multipart_upload.multipart.XMLMPUContainer")
    def test_upload_chunks_concurrently_from_bytesio_bad_request(
            self, mock_container):
        mock_container.upload_id = "321"
        mock_blob = mock.Mock(Blob)
        mock_client = mock.Mock(spec=Client)
        mock_bucket = mock.Mock()
        mock_request = mock.Mock(request.Request)
        mock_request.text = "<Error><Code>TestCode</Code><Message>TestMessage</Message><Details>TestDetails</Details></Error>"
        mock_request.status_code = 400
        mock_request.headers = {"etag": 12345}
        mock_client._http.request.return_value = mock_request
        mock_bucket.name = "test_bucket"
        fake_metadata = {"name": "test/test-multipart-upload"}
        mock_blob._get_upload_arguments.return_value = [{
            "fake_header": "a"
        }, fake_metadata, "json"]
        mock_blob.name = "test_blob"
        mock_blob.kms_key_name = None
        mock_blob.client = mock_client
        mock_blob.bucket = mock_bucket
        test_bytes = io.BytesIO(b'12345' * 10000)
        self.assertRaises(
            InvalidResponse,
            multipart.upload_chunks_concurrently_from_bytesio,
            test_bytes,
            mock_blob,
            checksum=None,
        )

    def test_prepare_upload_request_finished(self):
        bvx = multipart._BufferViewXMLMPUPart(
            "www.testval.test",
            1234,
            b"abcdefghijklmnopqrstuvwxyz",
            0,
            3,
            1,
        )
        bvx._finished = True
        self.assertRaises(ValueError, bvx._prepare_upload_request)

    def test_prepare_upload_request(self):
        bvx = multipart._BufferViewXMLMPUPart("www.testval.test",
                                              1234,
                                              b"abcdefghijklmnopqrstuvwxyz",
                                              0,
                                              3,
                                              1,
                                              headers={"test": "one"})
        req_type, url, payload, headers = bvx._prepare_upload_request()
        self.assertEqual(req_type, "PUT")
        self.assertEqual(url, "www.testval.test?partNumber=1&uploadId=1234")
        self.assertEqual(payload, b'abc')
        self.assertEqual(headers, {"test": "one"})

    @mock.patch("dataflux_pytorch.multipart_upload.multipart.XMLMPUPart")
    def test_buffer_view_upload_part_success(self, mock_part):
        mock_client = mock.Mock(spec=Client)
        mock_request = mock.Mock(request.Request)
        want_etag = 12345
        want_part_num = 123
        mock_request.status_code = 200
        mock_client._http.request.return_value = mock_request
        mock_request.headers = {"etag": want_etag}
        got_part_num, got_etag = multipart._buffer_view_upload_part(
            mock_client,
            "www.testval.test",
            want_part_num,
            b"abcdefghijklmnopqrstuvwxyz",
            0,
            3,
            123,
            None,
            {"etag": want_etag},
            None,
        )
        self.assertEqual(got_part_num, want_part_num)
        self.assertEqual(got_etag, want_etag)

    @mock.patch("dataflux_pytorch.multipart_upload.multipart.XMLMPUPart")
    def test_buffer_view_upload_part_failure(self, mock_part):
        mock_client = mock.Mock(spec=Client)
        mock_request = mock.Mock(request.Request)
        mock_request.text = "<Error><Code>TestCode</Code><Message>TestMessage</Message><Details>TestDetails</Details></Error>"
        mock_request.status_code = 400
        mock_client._http.request.return_value = mock_request
        mock_request.headers = {"etag": 12345}
        self.assertRaises(
            InvalidResponse,
            multipart._buffer_view_upload_part,
            mock_client,
            "www.testval.test",
            1234,
            b"abcdefghijklmnopqrstuvwxyz",
            0,
            3,
            123,
            None,
            {"etag": 12345},
            None,
        )
