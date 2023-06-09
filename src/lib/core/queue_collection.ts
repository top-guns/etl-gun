import { CollectionOptions } from "./base_collection.js";
import { BaseCollection_ID, BaseCollection_ID_Event } from "./base_collection_id.js";
import { BaseEndpoint } from "./endpoint.js";

export type BaseQueueCollectionEvent = BaseCollection_ID_Event;

export abstract class BaseQueueCollection<T> extends BaseCollection_ID<T> {
}
  