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

def readref_tree(article_name, citation):
    print("At: ["+article_name+"]")
    ## im assuming if it knows the citing article, it has its name
    article = scholarly.search_pubs_query(article_name)
    try:
        article = next(article).fill()
        if article.get_citedby() is not None:
            for citation in article.get_citedby():
                readref_tree(citation.bib['title'], article_name)
        else:
            for author in article.bib['author'].split('and'):
                query = scholarly.search_author(author)
                try:
                    this = next(query)
                    print(article_name + ',' + author + ',' + this.affiliation + ',' + citation)
                except StopIteration:
                    print(article_name + ',' + author + ',Unknown affiliation,' + citation)
    except:
        print(article_name + ',Request not completed,Unknown affiliation,Request not completed')

readref_tree('Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture','')
