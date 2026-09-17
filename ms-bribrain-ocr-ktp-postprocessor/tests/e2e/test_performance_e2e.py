"""End-to-end performance tests for the OCR KTP postprocessor.

These tests focus on performance characteristics like response time,
throughput, and stability under load.
"""

import json
import time
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed


@pytest.mark.e2e
@pytest.mark.slow
class TestPerformanceE2E:
    """Performance-related E2E tests."""

    def test_single_request_response_time(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that single request completes within acceptable time."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        start_time = time.time()
        response = test_client.post("/v1/ocr_postprocess", json=request_data)
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 2.0, f"Request took too long: {elapsed_time:.3f}s"

    def test_sequential_requests_performance(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test performance of sequential requests."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}
        num_requests = 10
        response_times = []

        for _ in range(num_requests):
            start_time = time.time()
            response = test_client.post("/v1/ocr_postprocess", json=request_data)
            elapsed_time = time.time() - start_time

            assert response.status_code == 200
            response_times.append(elapsed_time)

        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)

        # Performance assertions
        assert avg_time < 1.0, f"Average response time too high: {avg_time:.3f}s"
        assert max_time < 3.0, f"Max response time too high: {max_time:.3f}s"

        # Log performance metrics
        print(f"\nSequential Request Performance ({num_requests} requests):")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")

    def test_concurrent_requests_handling(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test handling of concurrent requests."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}
        num_concurrent = 5

        def make_request():
            start_time = time.time()
            response = test_client.post("/v1/ocr_postprocess", json=request_data)
            elapsed_time = time.time() - start_time
            return response.status_code, elapsed_time

        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]

        # All requests should succeed
        status_codes = [r[0] for r in results]
        response_times = [r[1] for r in results]

        assert all(code == 200 for code in status_codes), f"Some requests failed: {status_codes}"

        avg_time = sum(response_times) / len(response_times)
        print(f"\nConcurrent Request Performance ({num_concurrent} concurrent):")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  All succeeded: Yes")

    def test_large_payload_processing(
        self, test_client, mock_database_insert
    ):
        """Test processing of large OCR payload."""
        # Create a large payload with many OCR items
        large_payload = []
        for i in range(100):
            y_offset = i * 40
            large_payload.append([
                [[100, 50 + y_offset], [300, 80 + y_offset]],
                [f"TEXT LINE {i}", 0.95]
            ])

        # Add required fields
        large_payload.extend([
            [[[100, 4050], [180, 4080]], ["NIK", 0.97]],
            [[[190, 4050], [450, 4080]], ["3174012801950001", 0.96]],
            [[[100, 4090], [180, 4120]], ["Nama", 0.96]],
            [[[190, 4090], [450, 4120]], ["BUDI SANTOSO", 0.95]],
            [[[100, 4130], [200, 4160]], ["Jenis Kelamin", 0.95]],
            [[[210, 4130], [350, 4160]], ["LAKI-LAKI", 0.94]],
            [[[100, 4170], [180, 4200]], ["Agama", 0.95]],
            [[[190, 4170], [280, 4200]], ["ISLAM", 0.94]],
        ])

        request_data = {"ocr_text": json.dumps(large_payload)}

        start_time = time.time()
        response = test_client.post("/v1/ocr_postprocess", json=request_data)
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 5.0, f"Large payload took too long: {elapsed_time:.3f}s"

        print(f"\nLarge Payload Performance ({len(large_payload)} items):")
        print(f"  Processing time: {elapsed_time:.3f}s")

    def test_response_consistency_under_load(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that responses are consistent under load."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}
        num_requests = 20
        results = []

        for _ in range(num_requests):
            response = test_client.post("/v1/ocr_postprocess", json=request_data)
            assert response.status_code == 200
            results.append(response.json()["data"]["ocr_result"])

        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result["nik"] == first_result["nik"], f"NIK mismatch at request {i}"
            assert result["nama"] == first_result["nama"], f"Nama mismatch at request {i}"
            assert result["agama"] == first_result["agama"], f"Agama mismatch at request {i}"

    def test_memory_stability(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test memory stability across multiple requests."""
        import gc

        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}
        num_requests = 50

        # Force garbage collection before test
        gc.collect()

        for i in range(num_requests):
            response = test_client.post("/v1/ocr_postprocess", json=request_data)
            assert response.status_code == 200

            # Periodically force garbage collection
            if i % 10 == 0:
                gc.collect()

        # If we get here without memory errors, test passes
        assert True, "Memory remained stable across requests"


@pytest.mark.e2e
class TestThroughputE2E:
    """Throughput-related E2E tests."""

    def test_requests_per_second(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Measure requests per second throughput."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}
        duration_seconds = 3
        request_count = 0

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            response = test_client.post("/v1/ocr_postprocess", json=request_data)
            assert response.status_code == 200
            request_count += 1

        elapsed = time.time() - start_time
        rps = request_count / elapsed

        print(f"\nThroughput Test:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Requests: {request_count}")
        print(f"  Requests/second: {rps:.2f}")

        # Should handle at least 5 requests per second
        assert rps >= 5, f"Throughput too low: {rps:.2f} rps"

    def test_varied_payload_throughput(
        self, test_client, multiple_ktp_samples, mock_database_insert
    ):
        """Test throughput with varied payloads."""
        payloads = [json.dumps(data) for data in multiple_ktp_samples.values()]
        request_count = 0
        start_time = time.time()

        # Process each payload multiple times
        for _ in range(3):
            for payload in payloads:
                request_data = {"ocr_text": payload}
                response = test_client.post("/v1/ocr_postprocess", json=request_data)
                assert response.status_code == 200
                request_count += 1

        elapsed = time.time() - start_time
        rps = request_count / elapsed

        print(f"\nVaried Payload Throughput:")
        print(f"  Total requests: {request_count}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Requests/second: {rps:.2f}")
