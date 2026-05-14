import json
import logging

import azure.functions as func

from .orchestrator import Orchestrator

bp_2 = func.Blueprint()

# TODO: the event grid subscription should filter by subject
@bp_2.function_name(name="indexer_pokeseriex_eventgrid")
@bp_2.event_grid_trigger(arg_name="event")
def main(event: func.EventGridEvent):
    try:
        result = json.dumps({
            'id': event.id,
            'data': event.get_json(),
            'topic': event.topic,
            'subject': event.subject,
            'event_type': event.event_type,
        })
        logging.info('Python EventGrid trigger processed an event: %s', result)
    except Exception as e:
        logging.error("Error processing event data: %s", str(e))
        return

    index_name = Orchestrator.get_index_name()

    if index_name in event.subject:
        index_client, search_client = Orchestrator.create_search_clients(index_name)
        try:
            Orchestrator.check_index(index_name, index_client=index_client)
        except Exception as e:
            logging.error("Error checking index '%s': %s", index_name, str(e))

        if event.event_type == "Microsoft.Storage.BlobCreated":
            logging.info("Blob Created: %s", event.subject)
            try:
                Orchestrator.document_indexing(index_name, event.subject, search_client=search_client)
            except Exception as e:
                logging.error("Error indexing document from blob '%s': %s", event.subject, str(e))

        elif event.event_type == "Microsoft.Storage.BlobDeleted":
            logging.info("Blob deleted: %s", event.subject)
            try:
                Orchestrator.delete_document_by_filename(index_name, event.subject, search_client=search_client)
            except Exception as e:
                logging.error("Error unindexing document from blob '%s': %s", event.subject, str(e))
