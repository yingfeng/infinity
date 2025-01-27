// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module;

#include "serialize.h"

export module serialize;

namespace infinity {

export using infinity::GetSizeInBytes;
export using infinity::ReadBuf;
export using infinity::ReadBufAdv;
export using infinity::ReadBufVecAdv;
export using infinity::WriteBuf;
export using infinity::WriteBufAdv;
export using infinity::WriteBufVecAdv;
export using infinity::GetSizeInBytesAligned;
export using infinity::GetSizeInBytesVecAligned;
export using infinity::ReadBufAdvAligned;
export using infinity::ReadBufVecAdvAligned;
export using infinity::WriteBufAdvAligned;
export using infinity::WriteBufVecAdvAligned;

} // namespace infinity