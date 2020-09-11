from ASAPPpy.sentences.classes import Sentence

s = Sentence("O João gosta de jogar futebol e de vez em quando também joga basquete em Viseu quando o tempo está bom")

print(s.text)
print(s.lemmas)

# s.compute_all()
s.compute_tags()
s.compute_lemmas()
s.compute_entities()

print(s.text)
print(s.lemmas)
print(s.tags)
print(s.entities)
