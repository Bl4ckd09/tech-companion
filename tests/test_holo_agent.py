from app.holo.agent import ClickTool, HoloStep, trim_to_last_images
from app.schemas import ClickAction


def observation(name: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "<observation>"},
            {"type": "image_url", "image_url": {"url": name}},
        ],
    }


def test_holo_click_converts_to_public_action() -> None:
    step = HoloStep(
        thought="Open Display.",
        tool_call=ClickTool(tool_name="click", element="Display", x=480, y=640),
    )
    action = step.to_action()
    assert action == ClickAction(element="Display", x=480, y=640)


def test_image_history_keeps_only_last_three_screenshots() -> None:
    messages = [observation(str(index)) for index in range(5)]
    trim_to_last_images(messages, keep=3)
    image_chunks = [
        chunk
        for message in messages
        for chunk in message["content"]
        if chunk["type"] == "image_url"
    ]
    evicted_chunks = [
        chunk
        for message in messages
        for chunk in message["content"]
        if chunk.get("text") == "[screenshot evicted]"
    ]
    assert len(image_chunks) == 3
    assert len(evicted_chunks) == 2
