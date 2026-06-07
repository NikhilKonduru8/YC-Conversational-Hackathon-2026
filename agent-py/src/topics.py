"""The Jarvis "ROS graph": topic + service names shared by all nodes.

Topics (pub/sub):
    /jarvis/state           current talking-sequence stage  -> display
    /jarvis/wake            wake-word fired {score}
    /jarvis/transcript      recognized user request (str)
    /jarvis/vision          Qwen-2.5-VL scene summary (str)
    /jarvis/context         Moss retrieval result {query, matches}
    /jarvis/response_chunk  a spoken sentence of the answer (str)  -> tts
    /jarvis/response        the full answer text (str), for logging
    /jarvis/interrupt       stop playback now (barge-in)

Services (request/response):
    audio/wait_wake   () -> score    block until "Hey Jarvis"
    audio/listen      () -> text     stream mic -> STT live, return the transcript
    vision/describe   ({transcript}) -> summary
    retrieval/compile ({transcript, vision}) -> grounding prompt
    reasoning/respond ({grounding, user_text}) -> full text  (streams chunks)
"""

from __future__ import annotations

# Topics
TOPIC_STATE = "/jarvis/state"
TOPIC_WAKE = "/jarvis/wake"
TOPIC_TRANSCRIPT = "/jarvis/transcript"
TOPIC_VISION = "/jarvis/vision"
TOPIC_CONTEXT = "/jarvis/context"
TOPIC_RESPONSE_CHUNK = "/jarvis/response_chunk"
TOPIC_RESPONSE = "/jarvis/response"
TOPIC_INTERRUPT = "/jarvis/interrupt"

# Services
SVC_WAIT_WAKE = "audio/wait_wake"
SVC_LISTEN = "audio/listen"  # stream mic -> STT live, return transcript
SVC_DESCRIBE = "vision/describe"
SVC_COMPILE = "retrieval/compile"
SVC_RESPOND = "reasoning/respond"
