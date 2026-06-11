/// Validate the big-endian i32 length field at `buf[1..5]` of a tagged
/// PostgreSQL wire message and return the exclusive end index of the message
/// (`1 + len`), or None if the message is malformed.
///
/// The wire length is a *signed* i32. Negative values are rejected before
/// widening — `as usize` would sign-extend a negative into a huge value,
/// after which the index math wraps and slice-indexing panics (process abort
/// across the LD_PRELOAD hook). All index arithmetic is checked so a crafted
/// length can never produce an out-of-bounds slice expression. The length
/// counts the 4-byte length field itself plus the payload, so the minimum
/// valid value is 5 (4 + 1 byte of payload, e.g. a lone null terminator).
fn checked_message_end(buf: &[u8]) -> Option<usize> {
    if buf.len() < 6 {
        return None;
    }
    let len = i32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]);
    if len < 0 {
        return None;
    }
    let len = len as usize;
    let end = 1usize.checked_add(len)?;
    if len < 5 || buf.len() < end {
        return None;
    }
    Some(end)
}

/// Extract SQL query text from a PostgreSQL wire protocol buffer.
/// Returns None if the buffer doesn't contain a recognizable query message.
pub fn extract_pg_query(buf: &[u8]) -> Option<&str> {
    if buf.is_empty() {
        return None;
    }
    match buf[0] {
        b'Q' => {
            // Simple query: 'Q' + i32 len + null-terminated SQL
            let end = checked_message_end(buf)?;
            let sql_bytes = &buf[5..end - 1]; // exclude null terminator
            std::str::from_utf8(sql_bytes).ok()
        }
        b'P' => {
            // Parse message: 'P' + i32 len + name(str0) + query(str0) + i16 nparams
            let end = checked_message_end(buf)?;
            let payload = &buf[5..end];
            // Skip statement name (null-terminated)
            let name_end = payload.iter().position(|&b| b == 0)?;
            let query_start = name_end + 1;
            let remaining = &payload[query_start..];
            let query_end = remaining.iter().position(|&b| b == 0)?;
            std::str::from_utf8(&remaining[..query_end]).ok()
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a Simple Query ('Q') message buffer.
    fn make_simple_query(sql: &str) -> Vec<u8> {
        // len = 4 (self) + sql.len() + 1 (null terminator)
        let len = (4 + sql.len() + 1) as u32;
        let mut buf = Vec::new();
        buf.push(b'Q');
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(sql.as_bytes());
        buf.push(0x00);
        buf
    }

    // Helper: build a Parse ('P') message buffer.
    fn make_parse_message(name: &str, query: &str) -> Vec<u8> {
        // payload = name + \0 + query + \0 + i16 nparams (2 bytes)
        // len = 4 (self) + name.len() + 1 + query.len() + 1 + 2
        let payload_len = name.len() + 1 + query.len() + 1 + 2;
        let len = (4 + payload_len) as u32;
        let mut buf = Vec::new();
        buf.push(b'P');
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.push(0x00);
        buf.extend_from_slice(query.as_bytes());
        buf.push(0x00);
        buf.extend_from_slice(&0u16.to_be_bytes()); // nparams = 0
        buf
    }

    #[test]
    fn test_empty_buffer() {
        assert_eq!(extract_pg_query(&[]), None);
    }

    #[test]
    fn test_non_query_message() {
        // 'B' is a Bind message — not handled
        let buf = vec![b'B', 0, 0, 0, 8, 0, 0, 0, 0];
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_simple_query_basic() {
        let sql = "SELECT 1";
        let buf = make_simple_query(sql);
        assert_eq!(extract_pg_query(&buf), Some(sql));
    }

    #[test]
    fn test_simple_query_select_table() {
        let sql = "SELECT * FROM users WHERE id = 1";
        let buf = make_simple_query(sql);
        assert_eq!(extract_pg_query(&buf), Some(sql));
    }

    #[test]
    fn test_simple_query_insert() {
        let sql = "INSERT INTO orders (user_id, amount) VALUES (42, 100)";
        let buf = make_simple_query(sql);
        assert_eq!(extract_pg_query(&buf), Some(sql));
    }

    #[test]
    fn test_simple_query_truncated_header() {
        // Only 5 bytes — not enough for a valid 'Q' message (need at least 6)
        let buf = vec![b'Q', 0, 0, 0, 9];
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_simple_query_truncated_body() {
        // Header says len=9, but total buffer is only 6 bytes (needs 1+9=10)
        let buf = vec![b'Q', 0, 0, 0, 9, b'S'];
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_simple_query_malformed_short_len() {
        // 'Q' message with len=4 (too short — no room for null terminator).
        // Previously panicked with slice start > end: buf[5..4].
        let buf = vec![b'Q', 0, 0, 0, 4, b'X', b'Y'];
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_simple_query_len_zero() {
        // 'Q' message with len=0 (clearly malformed).
        let buf = vec![b'Q', 0, 0, 0, 0, b'X', b'Y'];
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_simple_query_len_one() {
        // 'Q' message with len=1 (malformed).
        let buf = vec![b'Q', 0, 0, 0, 1, b'X', b'Y'];
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_simple_query_minimal() {
        // Minimal valid 'Q': empty SQL string — just the null terminator
        // len = 4 + 0 + 1 = 5
        let buf = vec![b'Q', 0, 0, 0, 5, 0x00];
        assert_eq!(extract_pg_query(&buf), Some(""));
    }

    #[test]
    fn test_simple_query_invalid_utf8() {
        // Build a message with invalid UTF-8 in the SQL body
        let len: u32 = 4 + 2 + 1; // 4 header + 2 body bytes + 1 null
        let mut buf = Vec::new();
        buf.push(b'Q');
        buf.extend_from_slice(&len.to_be_bytes());
        buf.push(0xff); // invalid UTF-8 byte
        buf.push(0xfe); // invalid UTF-8 byte
        buf.push(0x00); // null terminator
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_parse_message_unnamed_statement() {
        // Unnamed prepared statement (name = "")
        let sql = "SELECT id, name FROM products WHERE price > $1";
        let buf = make_parse_message("", sql);
        assert_eq!(extract_pg_query(&buf), Some(sql));
    }

    #[test]
    fn test_parse_message_named_statement() {
        let sql = "UPDATE accounts SET balance = $1 WHERE id = $2";
        let buf = make_parse_message("my_stmt", sql);
        assert_eq!(extract_pg_query(&buf), Some(sql));
    }

    #[test]
    fn test_parse_message_truncated_header() {
        // Only 5 bytes — not enough for a valid 'P' message (need at least 6)
        let buf = vec![b'P', 0, 0, 0, 8];
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_parse_message_truncated_body() {
        // Header says len=12, but buffer is too short
        let buf = vec![b'P', 0, 0, 0, 12, 0x00, b'S'];
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_parse_message_invalid_utf8_query() {
        // Construct a 'P' message where the query bytes are not valid UTF-8
        let name = "";
        // payload: name\0 + [0xff, 0xfe]\0 + 2 bytes nparams
        let payload_len = name.len() + 1 + 2 + 1 + 2;
        let len = (4 + payload_len) as u32;
        let mut buf = Vec::new();
        buf.push(b'P');
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.push(0x00); // end of name
        buf.push(0xff); // invalid UTF-8
        buf.push(0xfe); // invalid UTF-8
        buf.push(0x00); // end of query
        buf.extend_from_slice(&0u16.to_be_bytes()); // nparams
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_parse_message_missing_query_null_terminator() {
        // Payload has name\0 but no null terminator for the query
        let len: u32 = 4 + 1 + 3; // 4 header + 1 (name null) + 3 bytes with no query null
        let mut buf = Vec::new();
        buf.push(b'P');
        buf.extend_from_slice(&len.to_be_bytes());
        buf.push(0x00); // empty name
        buf.push(b'S');
        buf.push(b'E');
        buf.push(b'L'); // no null terminator for query
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_single_byte_buffer() {
        assert_eq!(extract_pg_query(&[b'Q']), None);
        assert_eq!(extract_pg_query(&[b'P']), None);
        assert_eq!(extract_pg_query(&[b'X']), None);
    }

    #[test]
    fn test_simple_query_negative_length_all_ff() {
        // 'Q' with length = 0xffffffff (i32 = -1). With the old `as usize`
        // sign-extension, len becomes a huge value; `1 + len` wraps in release
        // and the bounds check `buf.len() < 1 + len` passes, leading to a
        // slice panic on `buf[5..1 + len - 1]`. Must return None instead.
        let mut buf = vec![b'Q', 0xff, 0xff, 0xff, 0xff];
        buf.extend_from_slice(&[b'A'; 16]); // arbitrary trailing bytes
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_parse_negative_length_high_bit() {
        // 'P' with length = 0x80000000 (i32 = -2147483648, most negative).
        let mut buf = vec![b'P', 0x80, 0, 0, 0];
        buf.extend_from_slice(&[0u8; 16]);
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_simple_query_negative_length_minus_two() {
        // 'Q' with len = -2 (0xfffffffe). Crafted so that without checked
        // arithmetic the wrapped index math could pass bounds and slice-panic.
        let mut buf = vec![b'Q', 0xff, 0xff, 0xff, 0xfe];
        buf.extend_from_slice(&[b'B'; 16]);
        assert_eq!(extract_pg_query(&buf), None);
    }

    #[test]
    fn test_parse_negative_length_minus_one() {
        let mut buf = vec![b'P', 0xff, 0xff, 0xff, 0xff];
        buf.extend_from_slice(&[0u8; 16]);
        assert_eq!(extract_pg_query(&buf), None);
    }
}
