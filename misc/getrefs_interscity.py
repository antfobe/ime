#!/usr/bin/python

import scholarly
#search_query = scholarly.search_pubs_query('Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture')
#article = next(search_query).fill()
#for author in article.bib['author'].split('and'):
#    query = scholarly.search_author(author)
#    try:
#        this = next(query)
#        print(article + ',' + author + ',' + this.affiliation + ',')
#    except StopIteration:
#        print(article + ',' + author + ',Unknown affiliation,')

def readref_tree(article_name, citation, found):
    print("At: ["+article_name+"]")
    ## im assuming if it knows the citing article, it has its name
    article = scholarly.search_pubs_query(article_name)
    try:
        article = next(article).fill()
        if article.get_citedby() is not None and not any(article.get_citedby() in c for c in found['citations']):
            found['citations'].append(article.get_citedby())
            for ref in article.get_citedby():
                readref_tree(ref.bib['title'], article_name, found)
        else:
            for author in article.bib['author'].split('and'):
                if not any(author in a for a in found['authors']):
                    found['authors'].append(author)
                    query = scholarly.search_author(author)
                    try:
                        this = next(query)
                        with open('refsout.csv', 'a') as f:
                            f.write(article_name + ',' + author + ',' + this.affiliation + ',' + citation + '\n')
                    except StopIteration:
                        with open('refsout.csv', 'a') as f:
                            f.write(article_name + ',' + author + ',Unknown affiliation,' + citation + '\n')
    except:
        with open('refsout.csv', 'a') as f:
            f.write(article_name + ',Request not completed,Unknown affiliation,Request not completed' + '\n')

memdict = {'authors': [], 'citations': []}
readref_tree('Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture', '', memdict)
