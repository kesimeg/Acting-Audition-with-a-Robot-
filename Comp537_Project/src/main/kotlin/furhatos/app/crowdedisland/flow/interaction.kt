package furhatos.app.crowdedisland.flow

import furhatos.nlu.common.*
import furhatos.flow.kotlin.*
import furhatos.app.crowdedisland.nlu.*




import java.io.File





import com.eclipsesource.json.Json
import com.eclipsesource.json.JsonValue
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import furhatos.nlu.common.*
import furhatos.flow.kotlin.*

import furhatos.event.Event
import furhatos.gestures.Gestures
import furhatos.records.User
import furhatos.records.UserData
import netscape.javascript.JSObject




class start_record(
        val record : Boolean
) : Event("start_record")

class emotion_receive(
        val emotion : String
) : Event("emotion_receive")

val Start : State = state(Interaction) {

    onEntry {
        furhat.ask(" Hi my name is furhat. But as you already know I am one of the famous stage directors in the broadway. You are here for the acting audtion today right?")
    }

    onResponse<Yes>{
        goto(Instruction_state)
    }

    onResponse<No>{
        furhat.say("Ohh how unfortunate. But I feel that you got talent so I will get you in to audition anyways.")
        goto(Instruction_state)
    }
    onReentry {
        furhat.ask("You are here for the audition right?")
    }
}

val max_round = 4

val Instruction_state : State = state(Interaction) {

    onEntry {
        furhat.say("Alright then. Guess we can start. I will use a famous audition technique called Geilmans method. ")
        furhat.say("I will give a sentence and an emotion. You will repeat the sentence with the given emotion.")
        furhat.say("As an example I will give you the following sentence")
        delay(1000)
        random( {furhat.say("Do not disturb me and the emotion angry")
        furhat.say("You will do like this")
        delay(1000)
        furhat.gesture(Gestures.ExpressAnger, async = true)
        furhat.say("Do not disturb me!")
            delay(1000)
        }, {
            furhat.say("Its so rainy today and the emotion sad")
            furhat.say("You will do like this")
            delay(1000)
            furhat.gesture(Gestures.ExpressSad, async = true)
            furhat.say("Its so rainy today")
            delay(1000)
        },{
            furhat.say("What a good day and the emotion happy")
            furhat.say("You will do like this")
            delay(1000)
            furhat.gesture(Gestures.BigSmile, async = true)
            furhat.say("What a good day")
            delay(1000)
        })



        furhat.ask("Okay. I think it was clear enough. But I can explain again. Do you want me to explain again?")
    }

    onResponse<Yes>{
        reentry()

    }

    onResponse<No>{
        furhat.say("Okay. I will give you "+ max_round +" sentences. So you will perform "+max_round + "times. If you pass half of them you will get the role.")
        furhat.say("Also you must know that I will evaluate your performance based on your voice and your facial expression. So pay attention to this")
        goto(Game)
    }

    onResponse {
            reentry()
        }


    onReentry{
        furhat.say("I will give a sentence and an emotion. You will repeat the sentence with the given emotion.")
        furhat.say("As an example I will give you the following sentence")
        random( {furhat.say("Do not disturb me and the emotion angry")
            furhat.say("You will do like this")
            delay(1000)
            furhat.gesture(Gestures.ExpressAnger, async = true)
            furhat.say("Do not disturb me!")
            delay(1000)
        }, {
            furhat.say("Its so rainy today and the emotion sad")
            furhat.say("You will do like this")
            delay(1000)
            furhat.gesture(Gestures.ExpressSad, async = true)
            furhat.say("Its so rainy today")
            delay(1000)
        },{
            furhat.say("What a good day and the emotion happy")
            furhat.say("You will do like this")
            delay(1000)
            furhat.gesture(Gestures.BigSmile, async = true)
            furhat.say("What a good day")
            delay(1000)
        })
        furhat.ask("Do you want me to explain again?")

    }
}

var round_num = 0
var passed_round = 0

val sentence_list = listOf("Its eleven oclock","Its eleven oclock","Its eleven oclock","Its eleven oclock","Its eleven oclock") //sonra shuffle et
val emotions = listOf("angry","happy","sad")
var selected_emotion = 0

val Game : State = state(Interaction) {

    onEntry {



        if(round_num<4) {
            random({ furhat.say("Here comes round number " + (round_num + 1).toString()) },
                    { furhat.say("Okay now round " + (round_num + 1)).toString() },
                    { furhat.say("This is round " + (round_num + 1)).toString() })
            round_num++

            furhat.say("Your sentence is")
            delay(500)
            furhat.say(sentence_list[round_num % max_round])
            delay(100)

            selected_emotion = (0..3).random()

            furhat.say("Say this sentence with emotion " + emotions[selected_emotion])
            furhat.say("3 ")
            delay(200)
            send(start_record(true))
            furhat.say("2")
            delay(200)
            furhat.say("1")
            delay(200)
            furhat.say("Go!")
            delay(100)
        }
        else{
            furhat.say("Okay the audition is over")
            furhat.say("You got a score of "+ passed_round + " out of "+ max_round +" rounds")

            if(passed_round>=max_round/2){
                furhat.say("Congratulations you performed very good. You got the role")
                furhat.say("See you again")
            }
            else{
                furhat.say("Sorry you couldn't get enough points to get the role. But I feel like you got hidden talent. You can take the audition again anytime you want")
                furhat.say("See you again then")
            }

            goto(Idle)
        }
    }
    onEvent("emotion_receive"){
        println(it.getString("emotion"))
        if(emotions[selected_emotion] == it.getString("emotion")){
            furhat.say("Congratulations you performed well. You got a point.")
            passed_round++
        }
        else {
            furhat.say("Hmm this looked more like " + it.getString("emotion"))

            if (emotions[selected_emotion] == "angry") {
                furhat.say("To perform emotion angry try to talk loud or shout with and squeeze your eyebrows ")
            } else if (emotions[selected_emotion] == "happy") {
                furhat.say("To perform emotion happy smile and talk with regular voice level")
            } else if (emotions[selected_emotion] == "sad") {
                furhat.say("To perform emotion sad talk quietly")
            }
        }
        reentry()
    }
}

