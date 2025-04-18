import pyodbc
import logging
import uuid
from datetime import date

class Tasklist(object):

    def __init__(self, config):
        self.connection_string = config['connection_str']
        self.conn = None

    def update_tasks(self, docid, task_status):
        items = []
        logging.info(docid)
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                for task in task_status:
                    logging.info(task)
                    title = task["title"]
                    completed = task["completed"]
                    update_query = f"""
                        update [dbo].[document_tasks]
                        set completed = {completed}
                        where
                        docid = '{docid}' and
                        taskid = (select Id from [dbo].[ToDo] where title = '{title}');
                    """
                    logging.info(update_query)
                    cursor.execute(update_query)

            with self.get_conn() as conn:
                cursor = conn.cursor()
                select_query = f"""
                                select td.title, dt.completed from [dbo].[document_tasks] AS dt INNER JOIN [dbo].[ToDo] AS td
                                ON dt.taskid = td.Id
                                WHERE docid = '{docid}'
                                ORDER BY td.[order];
                                """
                cursor.execute(select_query);
                rows = cursor.fetchall()
                print(len(rows))
                if len(rows) > 0:
                    for row in rows:
                        row_dict = dict(zip([column[0] for column in cursor.description], row))
                        items.append(row_dict)

        except Exception as ex:
            print(ex)

        return items

    def get_tasks(self, docid):
        items = []
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                select_query = f"""
                                select td.title, dt.completed from [dbo].[document_tasks] AS dt INNER JOIN [dbo].[ToDo] AS td
                                ON dt.taskid = td.Id
                                WHERE docid = '{docid}'
                                ORDER BY td.[order];
                                """
                cursor.execute(select_query);
                rows = cursor.fetchall()
                # items.append(len(rows))
                if len(rows) > 0:
                    for row in rows:
                        row_dict = dict(zip([column[0] for column in cursor.description], row))
                        items.append(row_dict)
                else:
                    cursor.execute("SELECT Id FROM [dbo].[ToDo] ORDER BY [order];")
                    tasks = cursor.fetchall()
                    for task in tasks:
                        taskid = task[0]
                        cursor.execute("INSERT INTO [dbo].[document_tasks](docid, taskid, completed) VALUES(?, ?, ?);", (docid, taskid, 0))

                    cursor.execute(select_query)
                    rows = cursor.fetchall()
                    if len(rows) > 0:
                        for row in rows:
                            row_dict = dict(zip([column[0] for column in cursor.description], row))
                            items.append(row_dict)
        except Exception as ex:
            print(ex)
        
        return items
    
    def add_docs(self, docid, documents):
        with self.get_conn() as conn:
            cursor = conn.cursor() 
            for document in documents:
                this_id = uuid.uuid4()
                cursor.execute("INSERT INTO [dbo].[document_resources](id, docid, doc_path, doc_type) VALUES(?, ?, ?, ?);", (this_id, docid, document['filepath'], document['type']))


    def get_docs(self, docid):
        select_query = f"""
                        select doc_path, doc_type from [dbo].[document_resources]
                        WHERE docid = '{docid}';
                        """
        with self.get_conn() as conn:
            items = []
            cursor = conn.cursor() 
            cursor.execute(select_query)
            rows = cursor.fetchall()
            if len(rows) > 0:
                for row in rows:
                    row_dict = dict(zip([column[0] for column in cursor.description], row))
                    items.append(row_dict)
            
            logging.info(items)
            return items
        
    def update_doc_summary(self, docid, docpath, summary):
        update_query = f"""
                        update [dbo].[document_resources]
                        set summary = ?
                        where
                        docid = ?
                        and doc_path= ?;
                    """
        with self.get_conn() as conn:
            cursor = conn.cursor() 
            cursor.execute(update_query, (summary, docid, docpath))

    def get_doc_summary(self, docid, docpath):
        select_query = f"""
                        select summary
                        from [dbo].[document_resources]
                        where
                        docid = '{docid}'
                        and doc_path= '{docpath}';
                    """
        with self.get_conn() as conn:
            cursor = conn.cursor() 
            cursor.execute(select_query)
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return None
            
    def delete_document_tasks(self, docid):
        delete_query = f"""
            delete from [dbo].[document_tasks] where docid = '{docid}';
        """
        try:
             with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(delete_query)
        except Exception as ex:
            print(ex)

    def delete_document_resources(self, docid):
        delete_query = f"""
            delete from [dbo].[document_resources] where docid = '{docid}';
        """
        try:
             with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(delete_query)
        except Exception as ex:
            print(ex)

    def add_document_drafts(self, docid):
        current_date = date.today().strftime("%Y-%m-%d")
        insert_query = f"""
            insert into document_drafts (docid, [start_date], end_date, status) 
            values('{docid}', '{current_date}', null, 0);

        """
        try:
             with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(insert_query)
        except Exception as ex:
            raise(ex)

    def complete_document_draft(self, docid):
        current_date = date.today().strftime("%Y-%m-%d")
        update_query = f"""
            update document_drafts 
            set end_date = '{current_date}',
            status = 1
            where docid = '{docid}';
        """
        try:
             with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(update_query)
        except Exception as ex:
            print(ex)

    def get_document_draft_status(self, docid):
        select_query = f"""
            select docid, start_date, end_date, status from document_drafts 
            where docid = '{docid}';
        """
        try:
             with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(select_query)
                row = cursor.fetchone()
                if row:
                    row_dict = dict(zip([column[0] for column in cursor.description], row))
                    return row_dict
                else:
                    return None
                
        except Exception as ex:
            raise(ex)

        
    def get_conn(self):
        try:
            logging.info(self.connection_string)
            if not self.conn: 
                self.conn = pyodbc.connect(self.connection_string)
            
            return self.conn
        except Exception as ex:
            print(ex)
    
