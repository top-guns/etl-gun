import { BaseCollection_G_ID, BaseCollection_G_ID_Event } from "./base_collection_g_id.js";

export type QueueCollectionEvent = BaseCollection_G_ID_Event;
export abstract class QueueCollection<T> extends BaseCollection_G_ID<T> {
}
  