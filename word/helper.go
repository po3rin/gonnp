package word

// func containsInt(s []int, e int) bool {
// 	for _, v := range s {
// 		if e == v {
// 			return true
// 		}
// 	}
// 	return false
// }

func containsString(ss map[string]int, s string) bool {
	for v := range ss {
		if s == v {
			return true
		}
	}
	return false
}
