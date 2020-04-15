package furhatos.app.crowdedisland.nlu
import furhatos.nlu.EnumEntity
import furhatos.nlu.Intent
import furhatos.util.Language


class GoToForest: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("I would go discover the forest", "Discover the forest", "go to forest", "go discover the forest", "I would  rather go discover the forest", "forest")

    }}

class GoForest(var goToForest:GoToForest?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@goToForest")


    }}

class GoToBeach: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("I would stay on the beach", "stay on the beach", "go to the beach", "beach", "I would  rather stay on the beach")

    }}



class Gobeach(var beach:GoToBeach?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@beach")


    }}


class Follow: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("I would follow the animals", "Follow the animals", "Follow", "go follow animals", "I would  rather follow the animals.")

    }}

class Followanimals(var follow:Follow?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@follow")


    }}


class RunAway: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("I would run away", "Run away", "I run away.", "I would rather run away.")

    }}

class Runawayy(var runaway:RunAway?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@runaway")


    }}


class FollowRabbit: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("I would follow the rabbit", "I follow the rabbit", "follow the rabbit", "I would  rather follow the rabbit", "follow")

    }}

class Followrabbitt(var rabbit:FollowRabbit?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@rabbit")


    }}


class EatRabbit: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("I would eat the rabbit", "I eat the rabbit", "Eat the rabbit", "I would  rather eat the rabbit", "eat")

    }}

class Eatrabbitt(var rrabbit:EatRabbit?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@rrabbit")


    }}


class SleepinForest: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("Forest", "the forest", "I would sleep in the forest", "I sleep in the forest", "Sleep in the forest", "I would  rather sleep in the forest.")

    }}

class Sleepinforesst(var fforest:SleepinForest?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@fforest")


    }}

class SleepinBeach: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("Beach", "the beach","I would sleep in the beach", "I would sleep on the beach","I sleep in the beach","I sleep on the beach", "Sleep in the beach", "Sleep on the beach","I would  rather sleep on the beach.","I would  rather sleep in the beach." )

    }}

class Sleepinbbeach(var bbeach:SleepinBeach?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@bbeach")


    }}


class RuntotheBeach: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("I would run to the beach","I run to the beach", "Run to the beach", "beach","I would  rather run to the beach." )

    }}

class Runtobeach(var bbbeach:RuntotheBeach?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@bbbeach")


    }}


class HideintheForest: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("Forest", "the forest", "I would hide in the forest", "I hide in the forest", "Hide in the forest", "I would  rather hide in the forest.")

    }}

class Hideintheforessst(var ffforest:HideintheForest?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@ffforest")


    }}

class MakeFire: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("Fire", "make a fire", "I would make a fire", "I make a fire", "make a fire before sleeping", "I would  rather make a fire."
                ,"we would make a fire", "we make a fire", "we would  rather make a fire.", "we should make a fire", "we should make a fire before sleeping")

    }}

class Makefirrre(var fire:MakeFire?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@fire")


    }}


class NoFire: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("No fire", "don't make a fire", "I would not  make a fire", "I don't make a fire", "I would sleep without fire", "I would  sleep without fire.", "Sleep without fire", "I sleep without fire"
                , "We would not  make a fire", "We don't make a fire", "We would sleep without fire", "We should sleep without fire.", "we sleep without fire", "we would sleep without fire", "we should not make a fire.","We don't make fire.")

    }}

class Nofirrre(var ffire:NoFire?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@ffire")


    }}


class KeepExploring: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("Keep exploring", "Keep exploring the ship","Explore", "Explore the ship",  "I would keep exploring the ship", "I would explore ethe ship","I keep exploring the ship", "I explore the ship", "I would  rather keep exploring.", "I would rather explore the ship.","I would rather explore")

    }}

class Keepexploringgg(var ffforestt:KeepExploring?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@ffforestt")


    }}

class EatFish: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("Eat the sharks","eat shark","eat a shark","Eat the fish", "I would eat a shark", "we eat a shark", "we eat the fish", "we should eat the fish","we should eat a shark")

    }}

class Eatdafish(var fish:EatFish?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@fish")


    }}

class EatDead: EnumEntity(){
    override fun getEnum(lang: Language): List<String>{
        return listOf("Eat a dead body", "Eat the dead", "we should eat a dead body", "I eat a dead body", "I eat the dead body", "I would rather eat a dead body","I would rather eat the dead", "eat dead bodies", "we eat a dead body", "we eat the dead body", "we should rather eat a dead body","We should  rather eat the dead", "eat dead bodies")

    }}

class Eatledead(var fishh:EatDead?=null): Intent(){
    override fun getExamples(lang: Language): List<String>{
        return listOf("@fishh")


    }}