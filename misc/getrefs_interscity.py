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

iter = 0
def readref_tree(article_name, citation):
    print("starting iter: ["+str(iter)+"]")
    ## im assuming if it knows the citing article, it has its name
    article = next(scholarly.search_pubs_query(article_name)).fill()
    if article.get_citedby() is not None:
        for citation in article.get_citedby():
            readref_tree(citation.bib['title'], article_name)
    else:
        for author in article.bib['author'].split('and'):
            query = scholarly.search_author(author)
            try:
                this = next(query)
                print(article + ',' + author + ',' + this.affiliation + ',' + citation)
            except StopIteration:
                print(article + ',' + author + ',Unknown affiliation,' + citation)

readref_tree('Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture','')
