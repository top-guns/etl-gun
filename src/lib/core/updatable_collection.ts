import { BaseCollection_GF_IUD, BaseCollection_GF_IUD_Event } from "./base_collection_gf_iud.js";

export type UpdatableCollectionEvent = BaseCollection_GF_IUD_Event;
export abstract class UpdatableCollection<T> extends BaseCollection_GF_IUD<T> {
}
  