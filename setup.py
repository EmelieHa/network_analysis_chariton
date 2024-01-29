
from cltk import NLP
import re
   
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objects as go
import networkx as nx
import re




from lxml import etree

import networkx as nx
import matplotlib.pyplot as plt

from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
cltk_greek = NLP(language='grc', suppress_banner=True)

#create dictionary for all character-dictionaries
dict_of_dicts = {}



def divide_sections(text):
    tree = etree.parse(text)
    total_characters = []
    root = tree.getroot()
    for elem in root.findall('content'):
        """  book = elem.find('book').text
        chapter = elem.find('chapter').text """
        section = elem.find('text').text
        characters = find_characters(section)
        total_characters.append(characters)
    total_characters = ' '.join(total_characters)
    total_characters = word_tokenize(total_characters)
    total_set = set(total_characters)
    return total_set


def find_characters(section):
    characters_list = []
    non_speech = re.sub(r'(?<=«)(.*?)(?=»)', '', section)
    non_speech = ' '.join(non_speech)
    non_speech = word_tokenize(non_speech)
    non_speech_tokens = [w for w in non_speech if len(w) > 1]
    cltk_doc = cltk_greek.analyze(' '.join(non_speech_tokens))
    for sent in cltk_doc.sentences:
            for word in sent:
                if word.upos == 'PROPN':
                    characters_list.append(word.string)
    return ' '.join(characters_list)



  
#get full list of characters      
result_chariton = divide_sections('/Users/emeliehallenberg/cltk_data/De_Chaerea_et_Callirhoe.xml')
#print(result_chariton)


#make a list of result
full_characters_chariton = ['Κύπρου', 'Χαιρέαν', 'Θήρων', 'οὐδ', 'Ζῆθον', 'Στατείρας', 'Καλλιρόην·', 'Ἀρτέμιδι', 
                            'Πάτροκλον', 'Διόνυσος', 'Ἰταλίαν', 'Ἀσίας', 'Θήρωνος', 'Ἡρακλεῖ', 'Ἠ̣πείρου', 'Φθόνος',
                            'Πηλούσιον', 'Ἑρμοκράτει', 'Ἰωνίαν·', 'Ἄρειος', 'Φήμη·', 'Νηρηῒς', 'Ἀφροδίτην',
                            'Δημήτριος', 'Διονύσῳ', 'Ἔρως·', 'Θυμοίτην', 'Ὑγῖνον', 'Φοινίκην', 'Στάτειρα', 
                            'Σιωπῆς', 'Πλαγγών', 'Ἴωνα', 'Ἀφροδίτη', 'Λυδίας', 'Λεωνᾶ', 'Πολυχάρμῳ', 'Συρακοσία', 
                            'Ἀριάδνης', 'Συρίας', 'Πηλεῖ', 'Ἀφροδίτῃ·', 'Εὐφρᾶτα', 'Ῥοδογούνην', 'Μιλήτῳ', 
                            'Στάτειραν', 'Φαρνάκου', 'Αἴγυπτος', 'Ἀράδῳ', 'Ἀθρόον', 'Βαβυλῶνι', 'Μενέλεων',
                            'Ἰάσων', 'Βαβυλῶνα', 'Κῦρον', 'Σαλαμῖνι', 'Καρίας', 'Κύπρῳ', 'Χαιρέας', 'Ἄραδος',
                            'Ἄραδον', 'Θησέως', 'Ἀσίαν', 'Ἀρταξέρξης', 'Συρακούσας', 'Νιρέα', 'Διονύσιος·', 
                            'Ἀρταξάτην', 'Ὅμηρος', 'Νεῖλος', 'Ἀλκιβιάδην', 'Βαβυλῶνος', 'Ἀττικῆς', 'Συρίας·', 
                            'Τύχης', 'Διονύσιός', 'Ἡρακλέα', 'Πάφον', 'Τυρία', 'Χαιρέᾳ', 'Μιθριδάτης', 'Ἀράδου', 
                            'Ἔρωτα·', 'Καλλιρόην', 'Βαβυλών', 'Χαιρέα·', 'Διονύσιον·', 'Πλαγγόνι', 'Πριήνης', 
                            'Πρίαμον', 'Ζηλοτυπίαν', 'Ἰωνίᾳ', 'Συρία', 'Φαρνάκην', 'Φαρνάκῃ', 'Ἀρμενίας', 
                            'Ἐκβατάνοις', 'Σπάρτης', 'Σπάρτῃ', 'Διονύ-', 'ἀλλ', 'Σεμέλην', 'Εὐφράτην',
                            'Λιβύην·', 'Ζωπύρου', 'Θούριος', 'Αἴγυπτον', 'Σικελίας', 'Μιθριδάτου', 'Λήδας', 
                            'Παρθέν̣ο̣υ̣', 'Λεωνᾶν', 'Βάκτροις', 'Πλαταιαῖς', 'Πολύχαρμός', 'Πολυχάρμου', 'Στατείρᾳ',
                            'Τύρῳ', 'Κεφαλληνίᾳ', 'Φωκᾶν', 'Θήρωνα', 'Πλαγγόνος', 'Ἑρμοκράτην', 'Καλλιρόη·', 
                            'Ἱππόλυτον', 'Φαρνάκης', 'Καλλιρόης', 'Δωριεύς', '.', 'Καλλι⌝ρ̣ό̣ην', 'Καλλίσφυρον', 
                            'Ἄρτεμις', 'Μηδείας', 'Πολύχαρμον', 'Ἀφροδίτῃ', 'Βάκτρα', 'Σοῦσα', 'Ῥοδογούνη', 
                            'Μιθριδάτην', 'Εὐρώπης', 'Ἰταλιώτης', 'Τύρος', 'Κρήτης', 'Βάκτρων', 'Ξέρξην',
                            'Διονύσιος', 'Λευκώλενον', 'Κύρου', 'Ἀφροδίτης', 'Λεωνᾷ', 'Διονύσιε', 'Μιθριδάτῃ', 
                            'Πριήνῃ', 'μεθ', 'Τότ', 'Δημητρίου', 'Συρακοῦσαι', 'Ἀθηναγόρου', 'Νύμφης', 'Ἄρτεμιν', 
                            'Τύρον', 'Δημήτριον', 'Ἀρταξέρξῃ', 'Μίλητον', 'Διονύσιε·', 'Ἑρμοκράτης', 'Βαβυλὼν', 
                            'Σικελίαν', 'Θήρωνι', 'Σιδῶνος', 'Ἀΐδαο', 'Λυκίας', 'Σικελίᾳ', 'Ἀθηνᾶν', 'Ἀλκίνοον', 
                            'Κιλικίας', 'Ἰταλίας', 'Λεωνᾶς', 'Ἀφροδίτη·', 'Μέμφεως', 'Φοινίκη', 'Μίλητος', 'Χίον', 
                            'κατ', 'Χαιρέου', 'Ἀριάδνην', 'Ἀρτέμιδος', 'Ἔριν', 'Ἀθήνας', 'Θέτις', 'Ἀχιλλέως', 
                            'Συρακούσαις', 'Πάρις', 'Νισαίῳ', 'Διονυσίου', 'Ἔρωτος', 'Ἀρίστων', 'Καρίᾳ', 'Κρήτην·',
                            'Πολύχαρμος·', 'Σηρῶν', 'Μίλητός', 'Ἀλέξανδρος', 'Ἑλένην', 'Θερμοπύλαις', 'Βίας', 
                            'Πλαγγὼν', 'Πάνθοον', 'Ἴων', 'Εὐφράτου', 'Ἰωνίας', 'Χαρίτων', 'οὔτ', 'Καλλιρόῃ', 
                            'Ἄδραστος', 'Ὑγῖνος', 'Θέτιδος', 'Μενέλαος', 'Μεγαβύζου', 'Ἀφικομένη', 'Συρίαν', 
                            'Τυρίαν', 'Ζηνοφάνης', 'Φωκᾶς·', 'Διὸς', 'Μαραθῶνι', 'Μιλήτου', 'μετ', 'Φωκᾷ', 
                            'Πολύχαρμος', 'Συρακούσ̣αισ̣', 'Πλαγγόνα', 'Σικελίαν·', 'Νῦσος', 'Ζεύς·', 'Νέμεσιν',
                            'Ἑλλάδος', 'Κἀκείνη', 'Ἀχιλ̣⌜λέα', 'Ἰωνίαν', 'Διονυσίῳ', 'Καρίαν', 'Πηλίῳ', 'Χαιρέου·',
                            'Ἀρταξάτης', 'Ἑρμοκράτει·', 'Ἰωνία', 'Φωκᾶς', 'Ἐρύμανθον', ';', 'Φωκᾶν·', 'Ἀριάδνῃ', 
                            'Καλλιρόη', 'Χαιρέα', 'Τύρου', 'Χαιρέας·', 'Προτεσίλεως', 'Ἀρίστωνος', 'Δοκίμου', 
                            'Αἰγύπτου', 'Ἀναδραμὼν', 'Διονύσιον', 'Τηΰγετον', 'Βρασίδου', 'Συρακούσας·', 'Ἀμφίονα', 
                            'Ὑγίνῳ', 'Κοίλης', 'Διός']


#remove characters that are mentioned less than 5 times
def count_chars(text, li):
    tree = etree.parse(text)
    fulltext = ''
    root = tree.getroot()
    for elem in root.findall('content'):
        section = elem.find('text').text
        fulltext += section
    for char in li:
        sum = fulltext.count(char)
        if sum > 5:
            print(char)


#count_chars('/Users/emeliehallenberg/cltk_data/De_Chaerea_et_Callirhoe.xml', full_characters_chariton)

#manually create list of final characters with all name forms
final_list_chars = ['Χαιρέας', 'Χαιρέαν', 'Χαιρέᾳ', 'Χαιρέου', 'Χαιρέα',
'Θήρων', 'Θήρωνος', 'Θήρωνι',
'Στάτειρα', 'Στάτειραν', 'Στατείρᾳ',
'Πλαγγών', 'Πλαγγὼν', 'Πλαγγόνα',
'Λεωνᾶ', 'Λεωνᾶν', 'Λεωνᾶς',
'Ἀρταξάτης', 'Ἀρταξάτην',
'Μιθριδάτης', 'Μιθριδάτου', 'Μιθριδάτην', 'Μιθριδάτῃ',
'Καλλιρόην', 'Καλλιρόης', 'Καλλιρόῃ', 'Καλλιρόη',
'Πολυχάρμου', 'Πολύχαρμον', 'Πολύχαρμος', 'Πολυχάρμῳ',
'Ἑρμοκράτης', 'Ἑρμοκράτην', 'Ἑρμοκράτει',
'Ῥοδογούνη', 'Ῥοδογούνην',
'Διονύσιος', 'Διονύσιε', 'Διονυσίου', 'Διονυσίῳ', 'Διονύσιον',
'Φωκᾶς', 'Φωκᾶν', 'Φωκᾷ',
 ]

#add a dictionary for all name forms
for c in final_list_chars:
    dict_of_dicts[c] = {}


#clean sections from speech
def clean_sections(text):
    clean_sections = []
    tree = etree.parse(text)
    root = tree.getroot()
    for elem in root.findall('content'):
        section = elem.find('text').text
        non_speech = re.sub(r'(?<=«)(.*?)(?=»)', '', section)
        non_speech = word_tokenize(non_speech)
        non_speech_tokens = [w for w in non_speech if len(w) > 1]
        clean_sections.append(' '.join(non_speech_tokens))
    return clean_sections


sections_chariton = clean_sections('/Users/emeliehallenberg/cltk_data/De_Chaerea_et_Callirhoe.xml')
#print(sections_chariton)

#find co-occurrences in every section
def find_coocs(sections, char, li):
    char_dict = {}
    for char_2 in li:
        for sect in sections:
            if char_2 in sect and char in sect:
                if char_2 in char_dict:
                    char_dict[char_2] += 1
                else:
                    char_dict[char_2] = 1
                sect.replace(char, '')
    return char_dict
        


#iterate through every name form in the list, and add resulting dictionaries to the dict of dictionaries
for char in final_list_chars:
    dict_result = find_coocs(sections_chariton, char, final_list_chars)
    dict_of_dicts[char] = dict_result

""" for k, v in dict_of_dicts.items():
    print(k, v)  """  

#manually remove duplicates and add resulting co-occurrences together 
dict_of_dicts = {'Χαιρέας': {'Θήρων': 19, 'Πλαγγών': 3, 'Μιθριδάτης': 56, 'Καλλιρόη': 297, 
                             'Πολύχαρμος': 37, 'Ἑρμοκράτης': 26, 'Ῥοδογούνη': 8, 
                             'Διονύσιος': 98, 'Στάτειρα': 6, 'Φωκᾶς': 2},

'Θήρων': {'Χαιρέας': 19, 'Λεωνᾶ': 15,  'Καλλιρόη': 34, 'Πολύχαρμος': 1, 'Ἑρμοκράτης': 2, 'Διονύσιος': 8},

'Στάτειρα': {'Χαιρέας': 6,  'Μιθριδάτης': 3, 'Καλλιρόη': 45, 'Ῥοδογούνη': 4, 'Διονύσιος': 6, 'Ἀρταξάτης': 1},

'Πλαγγών': {'Καλλιρόη': 20, 'Διονύσιος': 19, 'Φωκᾶς': 1, 'Χαιρέας': 3, 'Λεωνᾶ': 4},
'Λεωνᾶ': {'Θήρων': 15, 'Πλαγγών': 4, 'Καλλιρόη': 21, 'Διονύσιος': 20, 'Φωκᾶς': 2},
'Ἀρταξάτης': {'Στάτειρα': 1, 'Καλλιρόη': 9, 'Πολύχαρμος': 1, 'Διονύσιος': 2},

'Μιθριδάτης': {'Χαιρέας': 56, 'Στάτειρα': 3, 'Καλλιρόη': 45, 'Πολύχαρμος': 10, 'Ἑρμοκράτης': 1, 'Διονύσιος': 27},
'Καλλιρόη': {'Χαιρέας': 297, 'Θήρων': 34, 'Στάτειρα': 45, 'Πλαγγών': 20, 'Λεωνᾶ': 21,  'Μιθριδάτης': 45, 
             'Πολύχαρμος': 21, 'Ἑρμοκράτης': 9, 'Διονύσιος': 124, 'Φωκᾶς': 9, 'Ῥοδογούνη': 8, 'Ἀρταξάτης': 9},
'Πολύχαρμος': {'Χαιρέας': 37,  'Μιθριδάτης': 10, 'Διονύσιος': 5, 'Καλλιρόη': 21, 'Θήρων': 1, 'Ἑρμοκράτης': 2, 'Ἀρταξάτης': 1},

'Ἑρμοκράτης': {'Χαιρέας': 26, 'Θήρων': 2, 'Μιθριδάτης': 1, 'Καλλιρόη': 9, 'Πολύχαρμος': 2, 'Διονύσιος': 3},

'Ῥοδογούνη': {'Χαιρέας': 8, 'Στάτειρα': 4, 'Καλλιρόη': 8, },

'Διονύσιος': {'Χαιρέας': 98, 'Θήρων': 8, 'Στάτειρα': 6, 'Πλαγγών': 19, 'Λεωνᾶ': 20, 'Μιθριδάτης': 27,
              'Καλλιρόη': 124, 'Πολύχαρμος': 5, 'Ἑρμοκράτης': 3, 'Φωκᾶς': 12, 'Ἀρταξάτης': 2},

'Φωκᾶς': {'Χαιρέας': 2, 'Καλλιρόη': 9, 'Διονύσιος': 12, 'Πλαγγών': 1, 'Λεωνᾶ': 2},
}


#create a dictionary for appearances of each character/name form
appearances = {}

def count_appearances(charlist, text):
    joined_text = ' '.join(text)
    for character in charlist:
        sum = joined_text.count(character)
        appearances[character] = sum

count_appearances(final_list_chars, sections_chariton)

#print(appearances)
        
#add up sums of appearances for each character
appearances = {'Χαιρέας': 443, 
               'Θήρων': 55,
               'Στάτειρα': 38,
               'Πλαγγών': 33,
               'Λεωνᾶ': 55,
                'Ἀρταξάτης': 10,
               'Μιθριδάτης': 69,
               'Καλλιρόη': 445,
            'Πολύχαρμος': 39, 
            'Ἑρμοκράτης': 30,
            'Ῥοδογούνη': 10,
            'Διονύσιος': 186,
            'Φωκᾶς': 12}

#create node for each character
call = nx.Graph()
for char in dict_of_dicts.keys():
    if appearances[char] > 0:
        call.add_node(char, size=appearances[char], color='orange')

#add edge for each co-occurrence
for char in dict_of_dicts.keys():
    for co_char in dict_of_dicts[char].keys():
        if dict_of_dicts[char][co_char] > 0:
            call.add_edge(char, co_char, weight=dict_of_dicts[char][co_char], text=str(dict_of_dicts[char][co_char]))

#get position for the nodes
pos_ = nx.spring_layout(call, seed=100)

#test from here:

def make_edge(x, y, text, width):
    
    '''Creates a scatter trace for the edge between x's and y's with given width

    Parameters
    ----------
    x    : a tuple of the endpoints' x-coordinates in the form, tuple([x0, x1, None])
    
    y    : a tuple of the endpoints' y-coordinates in the form, tuple([y0, y1, None])
    
    width: the width of the line

    Returns
    -------
    An edge trace that goes between x0 and x1 with specified width.
    '''
    return  go.Scatter(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = 'orangered'),
                       hoverinfo = 'text',
                       text      = ([text]),
                       mode      = 'lines')
    


# For each edge, make an edge_trace, append to list
edge_trace = []
for edge in call.edges():
    
    if call.edges()[edge]['weight'] > 0:
        char_1 = edge[0]
        char_2 = edge[1]

        x0, y0 = pos_[char_1]
        x1, y1 = pos_[char_2]

        text   = char_1 + '--' + char_2 + ': ' + str(call.edges()[edge]['weight'])
        
        trace  = make_edge([x0, x1, None], [y0, y1, None], text,
                           0.15*call.edges()[edge]['weight'])

        edge_trace.append(trace)
    
# Make a node trace
node_trace = go.Scatter(x         = [],
                        y         = [],
                        text      = [],
                        textposition = "middle center",
                        textfont_size = 15,
                        mode      = 'markers+text',
                        hoverinfo = 'none',
                        fillcolor='lightgreen',
                        marker = dict(color=['orange'], size=[])
)

# For each node get the position and size and add to the node_trace
for node in call.nodes():
    x, y = pos_[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['text'] += tuple(['<b>' + node + '</b> ']) 
    node_trace['marker']['size'] += tuple([call.nodes()[node]['size']*0.5])
    node_trace['marker']['color'] += tuple([call.nodes()[node]['color']])
    


layout = go.Layout(
    paper_bgcolor='rgba(0,200,100,0)',
    plot_bgcolor='rgba(0,0,100,0)'
)

fig = go.Figure(layout = layout)

for trace in edge_trace:
    fig.add_trace(trace)

fig.add_trace(node_trace)

fig.update_layout(showlegend = False)

fig.update_xaxes(showticklabels = False)

fig.update_yaxes(showticklabels = False)

fig.show()



#measure degree centrality
deg_cent = nx.degree_centrality(call)
""" for k, v in deg_cent.items():
    print(k, round(v, 2)) """
deg_bet = nx.betweenness_centrality(call)
print(deg_cent)
print(deg_bet)
