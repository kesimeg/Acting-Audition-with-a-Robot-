from threading import Thread
import time
import asyncio
import websockets
import json
import time


send_messagge = True
receive_messagge = False
message = None
import asyncio
import websockets

async def send_mess():
    uri = "ws://localhost:8080"
    while True:
        global send_messagge
        
        if send_messagge == True:
            message = '{"event_name":"Gesture","name":"Smile"}'
            async with websockets.connect(uri) as websocket:
                await websocket.send(message)
            send_messagge = False

async def receive_mess():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        while True:
            global receive_messagge
            global message
            message = await websocket.recv()
            receive_messagge = True



def thread1(threadname):
    #asyncio.get_event_loop().run_until_complete(hello())
    print()

def thread2(threadname):

    while True:
        global send_messagge
        global receive_messagge
        global message


        send_messagge = True

        if receive_messagge == True:
            print("thread",message)
            receive_messagge = False


#thread1 = Thread( target=thread1, args=("Thread-1", ) )
thread2 = Thread( target=thread2, args=("Thread-2", ) )

#thread1.start()
thread2.start()
#asyncio.get_event_loop().run_until_complete(send_mess())
asyncio.get_event_loop().run_until_complete(receive_mess())
#thread1.join()
thread2.join()
