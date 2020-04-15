package furhatos.app.crowdedisland.flow

import furhatos.flow.kotlin.*
import furhatos.util.*

val Idle: State = state {

    init {
        furhat.setVoice(Language.ENGLISH_US, Gender.MALE)
        if (users.count > 0) {
            furhat.attend(users.random)
            goto(Start)
        }
    }

    onEntry {
        furhat.attendNobody()
    }

    onUserEnter {
        furhat.attend(it)
        goto(Start)
    }
}

val myButton1 = Button("Angry")
val myButton2 = Button("Happy")
val myButton3 = Button("Sad")
val myButton4 = Button("Game")
val myButton5 = Button("Instruction_state")

val Interaction: State = state {

    onUserLeave(instant = true) {
        if (users.count > 0) {
            if (it == users.current) {
                furhat.attend(users.other)
                goto(Start)
            } else {
                furhat.glance(it)
            }
        } else {
            goto(Idle)
        }
    }

    onUserEnter(instant = true) {
        furhat.glance(it)
    }

    onButton(myButton1){
        raise(emotion_receive("angry"))}
    onButton(myButton2){
        raise(emotion_receive("happy"))}
    onButton(myButton3){
        raise(emotion_receive("sad"))}
    onButton(myButton4){
        goto(Game)
    }
    onButton(myButton5){
        goto(Instruction_state)
    }

}