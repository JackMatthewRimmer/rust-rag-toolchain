## What are the steps here ?

chunk text -> get embeddings back from openAI -> write them somewhere

should see this as a framework that is completely agnostic of source and destination

## how can we parallelize this without getting rate limited ?

should probably do the chunking in parallel collect the chunks and then send them of to openAI in some form of queue, while also taking advantage of being able to send batches to openAI