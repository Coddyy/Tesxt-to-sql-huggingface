import re
from langchain_community.utilities import SQLDatabase
import pyparsing
from dbconfig import DB_USERNAME, DB_PASSWORD, DB_HOST, DB_NAME, SCHEMA_NAME


# POSTGRES
db = SQLDatabase.from_uri("postgresql+psycopg2://{uname}:{password}@{host}/{db_name}".format(uname=DB_USERNAME, password=DB_PASSWORD, host=DB_HOST, db_name=DB_NAME))
def get_schema():
    schema = db.get_table_info()

    ## For removing comments
    comment = pyparsing.nestedExpr("/*", "*/").suppress()
    schema = comment.transformString(schema)

    # adding schema name
    if SCHEMA_NAME != None:
        schema = schema.replace('CREATE TABLE ', 'CREATE TABLE public.')
    # Remove newline
    schema = re.sub(r'(\n\s*)+\n+', '\n\n', schema)

    # f = open("schema.txt", "a")
    # f.write(schema)
    # f.close()
    return schema

# get_schema()