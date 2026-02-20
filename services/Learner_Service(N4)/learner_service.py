import sys
sys.path.append("../../stubs")  # adjust to your folder structure

import grpc
import buffer_pb2
import buffer_pb2_grpc
import random
import string


def random_string(length=4):
    return ''.join(random.choices(string.ascii_letters, k=length))

def test_buffer():
    """Test pushing transitions to N3 and sampling batches."""
    # Connect to Buffer Service (N3)
    channel = grpc.insecure_channel("localhost:50051")
    stub = buffer_pb2_grpc.BufferServiceStub(channel)

    # Push 50 random transitions
    for i in range(50):
        transition = buffer_pb2.Transition(
            state=random_string(),
            action=random_string(),
            reward=random.random(),
            next_state=random_string(),
            done=random.choice([True, False])
        )
        request = buffer_pb2.PushRequest(transition=transition)
        response = stub.PushTransition(request)
        print(f"Pushed transition {i+1}: status={response.ok}")

    # Sample 10 transitions
    sample_request = buffer_pb2.SampleRequest(batch_size=10)
    batch_response = stub.SampleBatch(sample_request)
    print("\nSampled batch of transitions:")
    for idx, t in enumerate(batch_response.transitions):
        print(f"{idx+1}: state={t.state}, action={t.action}, reward={t.reward}, done={t.done}")


if __name__ == "__main__":
    # Only run this test logic when executing directly
    test_buffer()