# Copyright 2025 Google LLC All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Firestore-backed AsyncKeyValue store for persistent OAuth client storage.

Used as client_storage for GoogleProvider so registered OAuth clients (e.g.
ChatGPT's cached client_id) survive Cloud Run instance restarts and scaling.
"""

import time
from collections.abc import Mapping, Sequence
from typing import Any

from google.cloud.firestore_v1 import AsyncClient


_DEFAULT_COLLECTION = "oauth_clients"
_EXPIRY_FIELD = "_expires_at"


class FirestoreKeyValue:
    """AsyncKeyValue protocol implementation backed by Cloud Firestore.

    Each KV collection maps to a Firestore sub-collection under a single
    top-level document, keeping all MCP client registrations isolated from
    other Firestore data in the project.
    """

    def __init__(self, client: AsyncClient, root_collection: str = "analytics_mcp_kv") -> None:
        self._client = client
        self._root = root_collection

    def _col(self, collection: str | None) -> Any:
        name = collection or _DEFAULT_COLLECTION
        return self._client.collection(self._root).document(name).collection("records")

    @staticmethod
    def _is_expired(data: dict) -> bool:
        expiry = data.get(_EXPIRY_FIELD)
        return expiry is not None and time.time() > expiry

    @staticmethod
    def _strip_meta(data: dict) -> dict:
        return {k: v for k, v in data.items() if k != _EXPIRY_FIELD}

    async def get(self, key: str, *, collection: str | None = None) -> dict[str, Any] | None:
        doc = await self._col(collection).document(key).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        if self._is_expired(data):
            await self.delete(key, collection=collection)
            return None
        return self._strip_meta(data)

    async def ttl(self, key: str, *, collection: str | None = None) -> tuple[dict[str, Any] | None, float | None]:
        doc = await self._col(collection).document(key).get()
        if not doc.exists:
            return None, None
        data = doc.to_dict()
        if self._is_expired(data):
            await self.delete(key, collection=collection)
            return None, None
        expiry = data.get(_EXPIRY_FIELD)
        remaining = (expiry - time.time()) if expiry is not None else None
        return self._strip_meta(data), remaining

    async def put(
        self,
        key: str,
        value: Mapping[str, Any],
        *,
        collection: str | None = None,
        ttl: float | None = None,
    ) -> None:
        payload = dict(value)
        if ttl is not None:
            payload[_EXPIRY_FIELD] = time.time() + float(ttl)
        await self._col(collection).document(key).set(payload)

    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        ref = self._col(collection).document(key)
        doc = await ref.get()
        if not doc.exists:
            return False
        await ref.delete()
        return True

    async def get_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> list[dict[str, Any] | None]:
        return [await self.get(k, collection=collection) for k in keys]

    async def ttl_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> list[tuple[dict[str, Any] | None, float | None]]:
        return [await self.ttl(k, collection=collection) for k in keys]

    async def put_many(
        self,
        keys: Sequence[str],
        values: Sequence[Mapping[str, Any]],
        *,
        collection: str | None = None,
        ttl: float | None = None,
    ) -> None:
        for key, value in zip(keys, values):
            await self.put(key, value, collection=collection, ttl=ttl)

    async def delete_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> int:
        count = 0
        for key in keys:
            if await self.delete(key, collection=collection):
                count += 1
        return count
