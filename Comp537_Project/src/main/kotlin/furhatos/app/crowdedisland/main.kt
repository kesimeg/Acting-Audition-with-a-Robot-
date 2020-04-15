package furhatos.app.crowdedisland

import furhatos.app.crowdedisland.flow.*
import furhatos.skills.Skill
import furhatos.flow.kotlin.*

class CrowdedislandSkill : Skill() {
    override fun start() {
        Flow().run(Idle)
    }
}

fun main(args: Array<String>) {
    Skill.main(args)
}
